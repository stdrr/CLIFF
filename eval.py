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

import torch
from torch.utils.data import DataLoader
import numpy as np
import cv2
import os
import argparse
import json
from collections import namedtuple, OrderedDict
from tqdm import tqdm
import torchgeometry as tgm
import re
import config
import common.constants as constants
import smplx
from datasets import BaseDataset
from utils.imutils import uncrop
from utils.pose_utils import reconstruction_error

from common.renderer_pyrd import Renderer
from lib.yolov3_detector import HumanDetector
from common.mocap_dataset import MocapDataset
from lib.yolov3_dataset import DetectionDataset
from models.cliff_hr48.cliff import CLIFF
from common.imutils import process_image

# from utils.part_utils import PartRenderer

# Define command-line arguments
parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint', default=None, help='Path to network checkpoint')
parser.add_argument('--dataset', default='h36m-p1', choices=['h36m-p1', 'h36m-p2', 'lsp', '3dpw', 'mpi-inf-3dhp', 'agora_test'], help='Choose evaluation dataset')
parser.add_argument('--log_freq', default=50, type=int, help='Frequency of printing intermediate results')
parser.add_argument('--batch_size', default=32, help='Batch size for testing')
parser.add_argument('--shuffle', default=False, action='store_true', help='Shuffle data')
parser.add_argument('--num_workers', default=8, type=int, help='Number of processes for data loading')
parser.add_argument('--result_file', default=None, help='If set, save detections to a .npz file')
parser.add_argument('--detection', default=False, action='store_true', help='If set, run detection before fitting')


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

    out_file = 'results/refit_bedlam_authors/pred_{chunk_number}.npz'.format(chunk_number=kwargs['chunk_number'])
    np.savez(out_file, imgname=imgnames, pred_vertices=pred_vertices_t, faces=kwargs['faces'])
    print('Saved chunk {chunk_number}'.format(chunk_number=kwargs['chunk_number']))


def run_detection(orig_img_bgr_all):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print("--------------------------- Detection ---------------------------")
    # Setup human detector
    human_detector = HumanDetector()
    det_batch_size = min(args.batch_size, len(orig_img_bgr_all))
    detection_dataset = DetectionDataset(orig_img_bgr_all, human_detector.in_dim)
    detection_data_loader = DataLoader(detection_dataset, batch_size=det_batch_size, num_workers=0)
    detection_all = []
    for batch_idx, batch in enumerate(tqdm(detection_data_loader)):
            
        norm_img = batch["norm_img"].to(device).float()
        dim = batch["dim"].to(device).float()

        detection_result = human_detector.detect_batch(norm_img, dim)
        detection_result[:, 0] += batch_idx * det_batch_size
        detection_all.extend(detection_result.cpu().numpy())
            
    detection_all = np.array(detection_all)
    return detection_all

    
def run_evaluation(model, dataset_name, dataset, result_file,
                   batch_size=32, img_res=224, 
                   num_workers=32, shuffle=False, log_freq=50):
    """Run evaluation on the datasets and metrics we report in the paper. """

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    use_gt_bbox = isinstance(dataset, BaseDataset)

    # Transfer model to the GPU
    print(model)
    model.to(device)

    # Load SMPL model
    SMPL = smplx.create
    smpl_neutral = SMPL(config.SMPL_MODEL_DIR,
                        create_transl=False).to(device)
    smpl_male = SMPL(config.SMPL_MODEL_DIR,
                     gender='male',
                     create_transl=False).to(device)
    smpl_female = SMPL(config.SMPL_MODEL_DIR,
                       gender='female',
                       create_transl=False).to(device)
    
    # renderer = PartRenderer()
    
    # Regressor for H36m joints
    J_regressor = torch.from_numpy(np.load(config.JOINT_REGRESSOR_H36M)).float()
    
    save_results = False
    # save_results = result_file is not None
    # Disable shuffling if you want to save the results
    if save_results:
        shuffle=False
    # Create dataloader for the dataset
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    
    # Pose metrics
    # MPJPE and Reconstruction error for the non-parametric and parametric shapes
    mpjpe = np.zeros(len(dataset))
    recon_err = np.zeros(len(dataset))
    mpjpe_smpl = np.zeros(len(dataset))
    recon_err_smpl = np.zeros(len(dataset))

    # Shape metrics
    # Mean per-vertex error
    shape_err = np.zeros(len(dataset))
    shape_err_smpl = np.zeros(len(dataset))

    # Mask and part metrics
    # Accuracy
    accuracy = 0.
    parts_accuracy = 0.
    # True positive, false positive and false negative
    tp = np.zeros((2,1))
    fp = np.zeros((2,1))
    fn = np.zeros((2,1))
    parts_tp = np.zeros((7,1))
    parts_fp = np.zeros((7,1))
    parts_fn = np.zeros((7,1))
    # Pixel count accumulators
    pixel_count = 0
    parts_pixel_count = 0

    # Store SMPL parameters
    smpl_pose = np.zeros((len(dataset), 72))
    smpl_betas = np.zeros((len(dataset), 10))
    smpl_camera = np.zeros((len(dataset), 3))
    pred_joints = np.zeros((len(dataset), 17, 3))

    eval_pose = True
    eval_masks = False
    eval_parts = False
    # Choose appropriate evaluation for each dataset
    if dataset_name == 'h36m-p1' or dataset_name == 'h36m-p2' or dataset_name == '3dpw' or dataset_name == 'mpi-inf-3dhp':
        eval_pose = True
    elif dataset_name == 'lsp':
        eval_masks = True
        eval_parts = True
        annot_path = config.DATASET_FOLDERS['upi-s1h']

    joint_mapper_h36m = constants.H36M_TO_J17 if dataset_name == 'mpi-inf-3dhp' else constants.H36M_TO_J14
    joint_mapper_gt = constants.J24_TO_J17 if dataset_name == 'mpi-inf-3dhp' else constants.J24_TO_J14
    
    # Define variables for volume
    imgnames = []
    pred_vertices_t = []
    chunk_number = 0
    
    # Iterate over the entire dataset
    for step, batch in enumerate(tqdm(data_loader, desc='Eval', total=len(data_loader))):
        # Get ground truth annotations from the batch
        # gt_pose = batch['pose'].to(device)
        # gt_betas = batch['betas'].to(device)
        # print(gt_betas.shape, gt_pose.shape)
        # gt_vertices = smpl_neutral(betas=gt_betas, body_pose=gt_pose[:, 3:], global_orient=gt_pose[:, :3]).vertices
        images = batch['img'].to(device)
        # gender = batch['gender'].to(device)
        # curr_batch_size = images.shape[0]
        batch_imgnames = batch['imgname']

        center = batch["center"].to(device).float()
        scale = batch["scale"].to(device).float()
        img_h = batch["img_h"].to(device).float()
        img_w = batch["img_w"].to(device).float()
        focal_length = batch["focal_length"].to(device).float()

        if use_gt_bbox:
            bbox_info = batch['bbox_info'].to(device)
        else:
            cx, cy, b = center[:, 0], center[:, 1], scale * 200
            bbox_info = torch.stack([cx - img_w / 2., cy - img_h / 2., b], dim=-1)
            bbox_info[:, :2] = bbox_info[:, :2] / focal_length.unsqueeze(-1) * 2.8  # [-1, 1]
            bbox_info[:, 2] = (bbox_info[:, 2] - 0.24 * focal_length) / (0.06 * focal_length)  # [-1, 1]
        full_img_shape = torch.stack((img_h, img_w), dim=-1)
        # print(bbox_info.shape)
        
        # if we are using the ground-truth information for the bbox, crop the images accordingly        
        if use_gt_bbox:
            images, _, _, _, _, _ = process_image(images, bbox_info)

        with torch.no_grad():
            pred_rotmat, pred_betas, pred_cam_crop = model(images, bbox_info)
            
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
        
    save_model_output(pred_vertices_t=pred_vertices_t, imgnames=imgnames, chunk_number=chunk_number, faces=smpl_neutral.faces)
    

    #     # ## for vis



    #     # with torch.no_grad():
    #     #     # pred_rotmat, pred_betas, pred_camera = model(images, bbox_info)
    #     #     pred_rotmat, pred_betas, pred_camera = model(images)
    #     #     pred_output = smpl_neutral(betas=pred_betas, body_pose=pred_rotmat[:,1:], global_orient=pred_rotmat[:,0].unsqueeze(1), pose2rot=False)
    #     #     pred_vertices = pred_output.vertices

    #     # if save_results:
    #     #     rot_pad = torch.tensor([0,0,1], dtype=torch.float32, device=device).view(1,3,1)
    #     #     rotmat = torch.cat((pred_rotmat.view(-1, 3, 3), rot_pad.expand(curr_batch_size * 24, -1, -1)), dim=-1)
    #     #     pred_pose = tgm.rotation_matrix_to_angle_axis(rotmat).contiguous().view(-1, 72)
    #     #     smpl_pose[step * batch_size:step * batch_size + curr_batch_size, :] = pred_pose.cpu().numpy()
    #     #     smpl_betas[step * batch_size:step * batch_size + curr_batch_size, :]  = pred_betas.cpu().numpy()
    #     #     smpl_camera[step * batch_size:step * batch_size + curr_batch_size, :]  = pred_camera.cpu().numpy()
            
    #     # 3D pose evaluation
    #     if eval_pose:
    #         # Regressor broadcasting
    #         J_regressor_batch = J_regressor[None, :].expand(pred_vertices.shape[0], -1, -1).to(device)
    #         # Get 14 ground truth joints
    #         if 'h36m' in dataset_name or 'mpi-inf' in dataset_name:
    #             gt_keypoints_3d = batch['pose_3d'].cuda()
    #             gt_keypoints_3d = gt_keypoints_3d[:, joint_mapper_gt, :-1]
    #         # For 3DPW get the 14 common joints from the rendered shape
    #         else:
    #             gt_vertices = smpl_male(global_orient=gt_pose[:,:3], body_pose=gt_pose[:,3:], betas=gt_betas).vertices 
    #             gt_vertices_female = smpl_female(global_orient=gt_pose[:,:3], body_pose=gt_pose[:,3:], betas=gt_betas).vertices 
    #             gt_vertices[gender==1, :, :] = gt_vertices_female[gender==1, :, :]
    #             gt_keypoints_3d = torch.matmul(J_regressor_batch, gt_vertices)
    #             gt_pelvis = gt_keypoints_3d[:, [0],:].clone()
    #             gt_keypoints_3d = gt_keypoints_3d[:, joint_mapper_h36m, :]
    #             gt_keypoints_3d = gt_keypoints_3d - gt_pelvis 


    #         # Get 14 predicted joints from the mesh
    #         pred_keypoints_3d = torch.matmul(J_regressor_batch, pred_vertices)
    #         if save_results:
    #             pred_joints[step * batch_size:step * batch_size + curr_batch_size, :, :]  = pred_keypoints_3d.cpu().numpy()
    #         pred_pelvis = pred_keypoints_3d[:, [0],:].clone()
    #         pred_keypoints_3d = pred_keypoints_3d[:, joint_mapper_h36m, :]
    #         pred_keypoints_3d = pred_keypoints_3d - pred_pelvis 

    #         # Absolute error (MPJPE)
    #         error = torch.sqrt(((pred_keypoints_3d - gt_keypoints_3d) ** 2).sum(dim=-1)).mean(dim=-1).cpu().numpy()
    #         mpjpe[step * batch_size:step * batch_size + curr_batch_size] = error

    #         # Reconstuction_error
    #         r_error = reconstruction_error(pred_keypoints_3d.cpu().numpy(), gt_keypoints_3d.cpu().numpy(), reduction=None)
    #         recon_err[step * batch_size:step * batch_size + curr_batch_size] = r_error


    #         # perspective projection 


    #     # # If mask or part evaluation, render the mask and part images
    #     # if eval_masks or eval_parts:
    #     #     mask, parts = renderer(pred_vertices, pred_camera)

    #     # # Mask evaluation (for LSP)
    #     # if eval_masks:
    #     #     center = batch['center'].cpu().numpy()
    #     #     scale = batch['scale'].cpu().numpy()
    #     #     # Dimensions of original image
    #     #     orig_shape = batch['orig_shape'].cpu().numpy()
    #     #     for i in range(curr_batch_size):
    #     #         # After rendering, convert imate back to original resolution
    #     #         pred_mask = uncrop(mask[i].cpu().numpy(), center[i], scale[i], orig_shape[i]) > 0
    #     #         # Load gt mask
    #     #         gt_mask = cv2.imread(os.path.join(annot_path, batch['maskname'][i]), 0) > 0
    #     #         # Evaluation consistent with the original UP-3D code
    #     #         accuracy += (gt_mask == pred_mask).sum()
    #     #         pixel_count += np.prod(np.array(gt_mask.shape))
    #     #         for c in range(2):
    #     #             cgt = gt_mask == c
    #     #             cpred = pred_mask == c
    #     #             tp[c] += (cgt & cpred).sum()
    #     #             fp[c] +=  (~cgt & cpred).sum()
    #     #             fn[c] +=  (cgt & ~cpred).sum()
    #     #         f1 = 2 * tp / (2 * tp + fp + fn)

    #     # # Part evaluation (for LSP)
    #     # if eval_parts:
    #     #     center = batch['center'].cpu().numpy()
    #     #     scale = batch['scale'].cpu().numpy()
    #     #     orig_shape = batch['orig_shape'].cpu().numpy()
    #     #     for i in range(curr_batch_size):
    #     #         pred_parts = uncrop(parts[i].cpu().numpy().astype(np.uint8), center[i], scale[i], orig_shape[i])
    #     #         # Load gt part segmentation
    #     #         gt_parts = cv2.imread(os.path.join(annot_path, batch['partname'][i]), 0)
    #     #         # Evaluation consistent with the original UP-3D code
    #     #         # 6 parts + background
    #     #         for c in range(7):
    #     #            cgt = gt_parts == c
    #     #            cpred = pred_parts == c
    #     #            cpred[gt_parts == 255] = 0
    #     #            parts_tp[c] += (cgt & cpred).sum()
    #     #            parts_fp[c] +=  (~cgt & cpred).sum()
    #     #            parts_fn[c] +=  (cgt & ~cpred).sum()
    #     #         gt_parts[gt_parts == 255] = 0
    #     #         pred_parts[pred_parts == 255] = 0
    #     #         parts_f1 = 2 * parts_tp / (2 * parts_tp + parts_fp + parts_fn)
    #     #         parts_accuracy += (gt_parts == pred_parts).sum()
    #     #         parts_pixel_count += np.prod(np.array(gt_parts.shape))

    #     # Print intermediate results during evaluation
    #     if step % log_freq == log_freq - 1:
    #         if eval_pose:
    #             print('MPJPE: ' + str(1000 * mpjpe[:step * batch_size].mean()))
    #             print('Reconstruction Error: ' + str(1000 * recon_err[:step * batch_size].mean()))
    #             print()
    #     #     if eval_masks:
    #     #         print('Accuracy: ', accuracy / pixel_count)
    #     #         print('F1: ', f1.mean())
    #     #         print()
    #     #     if eval_parts:
    #     #         print('Parts Accuracy: ', parts_accuracy / parts_pixel_count)
    #     #         print('Parts F1 (BG): ', parts_f1[[0,1,2,3,4,5,6]].mean())
    #     #         print()

    #     # if step % log_freq == log_freq - 1:
    #     #     print('MPJPE: ' + str(1000 * mpjpe.mean()))
    #     #     print('Reconstruction Error: ' + str(1000 * recon_err.mean()))
    #     #     print()

        
    # # Save reconstructions to a file for further processing
    # if save_results:
    #     np.savez(result_file, pred_joints=pred_joints, pose=smpl_pose, betas=smpl_betas, camera=smpl_camera)
    # # Print final results during evaluation
    # print('*** Final Results ***')
    # print()
    # if eval_pose:
    #     print('MPJPE: ' + str(1000 * mpjpe.mean()))
    #     print('Reconstruction Error: ' + str(1000 * recon_err.mean()))
    #     print()
    # # if eval_masks:
    # #     print('Accuracy: ', accuracy / pixel_count)
    # #     print('F1: ', f1.mean())
    # #     print()
    # # if eval_parts:
    # #     print('Parts Accuracy: ', parts_accuracy / parts_pixel_count)
    # #     print('Parts F1 (BG): ', parts_f1[[0,1,2,3,4,5,6]].mean())
    # #     print()

if __name__ == '__main__':
    args = parser.parse_args()
    model = CLIFF(config.SMPL_MEAN_PARAMS)
    checkpoint = torch.load(args.checkpoint)
    # model.load_state_dict(checkpoint['model'], strict=False)

    state_dict = checkpoint['model']

    revise_keys = [(r'^module\.', ''), ('encoder.', '')]

    # strip prefix of state_dict
    metadata = getattr(state_dict, '_metadata', OrderedDict())
    for p, r in revise_keys:
        state_dict = OrderedDict(
            {re.sub(p, r, k): v
             for k, v in state_dict.items()})
    # Keep metadata in state_dict
    state_dict._metadata = metadata

    print(model.state_dict().keys()) # , state_dict.keys()
    print('\n\n')
    print(state_dict.keys())


    # model.load_state_dict(checkpoint['model'], strict=False)
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    # Setup evaluation dataset
    dataset = BaseDataset(None, args.dataset, is_train=False)
    if args.detection:
        unique_imgs = np.unique(dataset.imgname)
        print("Loading images ...")
        orig_img_bgr_all = [cv2.imread(img_path) for img_path in tqdm(unique_imgs)]
        detection_all = run_detection(orig_img_bgr_all)
        dataset = MocapDataset(orig_img_bgr_all, detection_all, unique_imgs)
    # Run evaluation
    run_evaluation(model, args.dataset, dataset, args.result_file,
                   batch_size=args.batch_size,
                   shuffle=args.shuffle,
                   log_freq=args.log_freq)