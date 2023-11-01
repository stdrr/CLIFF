"""
This script can be used to evaluate a trained model on 3D pose/shape and masks/part segmentation. You first need to download the datasets and preprocess them.
Example usage:
```
python3 eval.py --checkpoint=data/model_checkpoint.pt --dataset=h36m-p1 --log_freq=20
```
Running the above command will compute the MPJPE and Reconstruction Error on the Human3.6M dataset (Protocol I). The ```--dataset``` option can take different values based on the type of evaluation you want to perform:
1. Human3.6M Protocol 1 ```--dataset=h36m-p1```
2. Human3.6M Protocol 2 ```--dataset=h36m-p2```
3. 3DPW ```--dataset=3dpw```
4. LSP ```--dataset=lsp```
5. MPI-INF-3DHP ```--dataset=mpi-inf-3dhp```
"""

import os
import torch
from torch.utils.data import DataLoader
import numpy as np
import cv2
import argparse
from tqdm import tqdm
import config
import smplx
from datasets import BaseDataset

from lib.yolov3_detector import HumanDetector
from common.mocap_dataset import MocapDataset
from lib.yolov3_dataset import DetectionDataset
from models.cliff_hr48.cliff import CLIFF
from common.imutils import process_image
from common.utils import strip_prefix_if_present


def cam_crop2full(crop_cam, center, scale, full_img_shape, focal_length):
    """
    convert the camera parameters from the crop camera to the full camera
    :param crop_cam: shape=(N, 3) weak perspective camera in cropped img coordinates (s, tx, ty)
    :param center: shape=(N, 2) bbox coordinates (c_x, c_y)
    :param scale: shape=(N, 1) square bbox resolution  (b / 200)
    :param full_img_shape: shape=(N, 2) original image height and width
    :param focal_length: shape=(N,)
    :return:
    """
    img_h, img_w = full_img_shape[:, 0], full_img_shape[:, 1]
    cx, cy, b = center[:, 0], center[:, 1], scale * 200
    w_2, h_2 = img_w / 2., img_h / 2.
    bs = b * crop_cam[:, 0] + 1e-9
    tz = 2 * focal_length / bs
    tx = (2 * (cx - w_2) / bs) + crop_cam[:, 1]
    ty = (2 * (cy - h_2) / bs) + crop_cam[:, 2]
    full_cam = torch.stack([tx, ty, tz], dim=-1)
    return full_cam


def save_model_output(**kwargs):
    pred_vertices_t = np.concatenate(kwargs['pred_vertices_t'])
    imgnames = np.concatenate(kwargs['imgnames'])

    out_file = 'results/agora_val/pred_detection_{chunk_number}.npz'.format(chunk_number=kwargs['chunk_number'])
    np.savez(out_file, imgname=imgnames, pred_vertices=pred_vertices_t, faces=kwargs['faces'], 
             detection_times=kwargs['detection_times'], estimation_times=kwargs['estimation_times'])
    print('Saved chunk {chunk_number}'.format(chunk_number=kwargs['chunk_number']))
    

def save_gt_verts(dataset):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    dataloader = DataLoader(dataset, batch_size=args.batch_size, num_workers=0)
    SMPL = smplx.create
    smpl_neutral = SMPL(config.SMPL_MODEL_DIR,
                        create_transl=False).to(device)
    imgs = []
    gt_verts = []
    for batch in tqdm(dataloader):
        gt_pose = batch['pose'].to(device)
        gt_betas = batch['betas'].to(device)
        gt_vertices = smpl_neutral(betas=gt_betas, body_pose=gt_pose[:, 3:], global_orient=gt_pose[:, :3]).vertices
        imgs.append(np.array(batch['imgname']))
        gt_verts.append(gt_vertices.cpu().numpy())
        
    imgs = np.concatenate(imgs)
    gt_verts = np.concatenate(gt_verts)
    out_file = 'results/agora_val/gt_verts.npz'
    np.savez(out_file, imgname=imgs, gt_vertices=gt_verts)
    print('Saved GT Vertices')


def run_detection(orig_img_bgr_all):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print("--------------------------- Detection ---------------------------")
    # Setup human detector
    human_detector = HumanDetector()
    det_batch_size = min(args.batch_size, len(orig_img_bgr_all))
    detection_dataset = DetectionDataset(orig_img_bgr_all, human_detector.in_dim)
    detection_data_loader = DataLoader(detection_dataset, batch_size=det_batch_size, num_workers=0)
    detection_all = []
    per_batch_times = []
    for batch_idx, batch in enumerate(tqdm(detection_data_loader)):
        
        start_batch = torch.cuda.Event(enable_timing=True)
        end_batch = torch.cuda.Event(enable_timing=True)
        start_batch.record()
            
        norm_img = batch["norm_img"].to(device).float()
        dim = batch["dim"].to(device).float()

        detection_result = human_detector.detect_batch(norm_img, dim)
        detection_result[:, 0] += batch_idx * det_batch_size
        detection_all.extend(detection_result.cpu().numpy())
        
        end_batch.record()
        # Waits for everything to finish running
        torch.cuda.synchronize()
        per_batch_times.append(start_batch.elapsed_time(end_batch))
            
    detection_all = np.array(detection_all)
    per_batch_times = np.array(per_batch_times)
    return detection_all, per_batch_times

    
def run_evaluation(model, dataset_name, dataset, result_file,
                   batch_size=32, img_res=224, 
                   num_workers=32, shuffle=False, log_freq=50, det_times=None):
    """Run evaluation on the datasets and metrics we report in the paper. """

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    use_gt_bbox = isinstance(dataset, BaseDataset)

    # Transfer model to the GPU
    model.to(device)

    # Load SMPL model
    SMPL = smplx.create
    smpl_neutral = SMPL(config.SMPL_MODEL_DIR,
                        create_transl=False).to(device)

    save_results = False
    # save_results = result_file is not None
    # Disable shuffling if you want to save the results
    if save_results:
        shuffle=False
    # Create dataloader for the dataset
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    
    # Define variables for volume
    imgnames = []
    pred_vertices_t = []
    chunk_number = 0
    per_batch_times = []
    # Iterate over the entire dataset
    for batch in tqdm(data_loader, desc='Eval', total=len(data_loader)):

        images = batch['img']
        batch_imgnames = batch['imgname']

        center = batch["center"].to(device).float()
        scale = batch["scale"].to(device).float()
        img_h = batch["img_h"].to(device).float()
        img_w = batch["img_w"].to(device).float()
        focal_length = batch["focal_length"].to(device).float()

        if use_gt_bbox:
            bbox_info = batch['bbox_info'].cpu().numpy()
        else:
            cx, cy, b = center[:, 0], center[:, 1], scale * 200
            bbox_info = torch.stack([cx - img_w / 2., cy - img_h / 2., b], dim=-1)
            bbox_info[:, :2] = bbox_info[:, :2] / focal_length.unsqueeze(-1) * 2.8  # [-1, 1]
            bbox_info[:, 2] = (bbox_info[:, 2] - 0.24 * focal_length) / (0.06 * focal_length)  # [-1, 1]
        
        full_img_shape = torch.stack((img_h, img_w), dim=-1)
        
        # if we are using the ground-truth information for the bbox, crop the images accordingly        
        if use_gt_bbox:
            images, _, _, _, _, _ = process_image(images, bbox_info)

        images = images.to(device)
        with torch.no_grad():
            start_batch = torch.cuda.Event(enable_timing=True)
            end_batch = torch.cuda.Event(enable_timing=True)
            start_batch.record()
            
            pred_rotmat, pred_betas, pred_cam_crop = model(images, bbox_info)
            
            end_batch.record()
            # Waits for everything to finish running
            torch.cuda.synchronize()
            per_batch_times.append(start_batch.elapsed_time(end_batch))
            
            pred_cam_full = cam_crop2full(pred_cam_crop, center, scale, full_img_shape, focal_length)
            pred_output = smpl_neutral(betas=pred_betas,
                                       body_pose=pred_rotmat[:, 1:],
                                       global_orient=pred_rotmat[:, [0]],
                                       pose2rot=False,
                                       transl=pred_cam_full)
            pred_vertices = pred_output.vertices

        # save output
        pred_vertices_t.append(pred_vertices.cpu().numpy())
        imgnames.append(batch_imgnames)
    
    per_batch_times = np.array(per_batch_times)
    save_model_output(pred_vertices_t=pred_vertices_t, imgnames=imgnames, chunk_number=chunk_number, faces=smpl_neutral.faces, 
                      detection_times=det_times, estimation_times=per_batch_times)
    return per_batch_times
    
    
def load_imgs(unique_imgs, debug):
    orig_img_bgr_all = []
    root_dir = '/media/hdd/stdrr/data/AGORA/images/validation_images_1280x720/validation_images_1280x720_{folder_idx}'
    unique_imgs = unique_imgs[:10,...] if debug else unique_imgs
    for imgname in tqdm(unique_imgs):
        folder_idx = imgname.split('/')[0].split('_')[-1]
        img_path = os.path.join(root_dir.format(folder_idx=folder_idx), imgname)
        img_bgr = cv2.imread(img_path)
        orig_img_bgr_all.append(img_bgr)
    return orig_img_bgr_all
    

if __name__ == '__main__':
    # Define command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', default=None, help='Path to network checkpoint')
    parser.add_argument('--dataset', default='h36m-p1', choices=['h36m-p1', 'h36m-p2', 'lsp', '3dpw', 'mpi-inf-3dhp', 'agora_validation'], help='Choose evaluation dataset')
    parser.add_argument('--log_freq', default=50, type=int, help='Frequency of printing intermediate results')
    parser.add_argument('--batch_size', default=32, help='Batch size for testing')
    parser.add_argument('--shuffle', default=False, action='store_true', help='Shuffle data')
    parser.add_argument('--num_workers', default=8, type=int, help='Number of processes for data loading')
    parser.add_argument('--result_file', default=None, help='If set, save detections to a .npz file')
    parser.add_argument('--detection', default=False, action='store_true', help='If set, run detection before fitting')
    parser.add_argument('--debug', default=False, action='store_true', help='If set, use debug mode')
    args = parser.parse_args()
    
    # Load the pretrained model
    print("Load the CLIFF checkpoint from path:", args.checkpoint)
    state_dict = torch.load(args.checkpoint, map_location=torch.device('cpu'))['model']
    state_dict = strip_prefix_if_present(state_dict, prefix="module.")
    cliff_model = CLIFF(config.SMPL_MEAN_PARAMS)
    cliff_model.load_state_dict(state_dict, strict=True)
    cliff_model.eval()

    # Setup evaluation dataset
    dataset = BaseDataset(None, args.dataset, is_train=False)
    # save_gt_verts(dataset)
    if args.detection:
        unique_imgs = np.unique(dataset.imgname)
        print("Loading images ...")
        orig_img_bgr_all = load_imgs(unique_imgs, args.debug)
        detection_all, per_batch_times_det = run_detection(orig_img_bgr_all)
        dataset = MocapDataset(orig_img_bgr_all, detection_all, unique_imgs)
    else:
        per_batch_times_det = np.array([0,0])
    # Run evaluation
    per_batch_times_eval = run_evaluation(cliff_model, args.dataset, dataset, args.result_file,
                                          batch_size=args.batch_size,
                                          shuffle=args.shuffle,
                                          log_freq=args.log_freq, det_times=per_batch_times_det)
    
    total_time = per_batch_times_det.mean() + per_batch_times_eval.mean()
    
    print(f'DETECTION TIME: {per_batch_times_det.mean()}')
    print(f'ESTIMATION TIME: {per_batch_times_eval.mean()}')
    print(f'TOTAL TIME: {total_time}')