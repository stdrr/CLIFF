# Copyright (C) 2022. Huawei Technologies Co., Ltd. All rights reserved.

# This program is free software; you can redistribute it and/or modify it
# under the terms of the MIT license.

# This program is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
# PARTICULAR PURPOSE. See the MIT License for more details.

import os
import os.path as osp
import cv2
import glob
import torch
import argparse
import numpy as np
from tqdm import tqdm
import smplx
from torch.utils.data import DataLoader
import torchgeometry as tgm
from models.cliff_hr48.cliff import CLIFF as cliff_hr48
from models.cliff_res50.cliff import CLIFF as cliff_res50
from common import constants
from common.utils import strip_prefix_if_present, cam_crop2full, video_to_images
from common.utils import estimate_focal_length
from common.renderer_pyrd import Renderer
from lib.yolov3_detector import HumanDetector
from common.mocap_dataset import MocapDataset
from lib.yolov3_dataset import DetectionDataset

from collections import defaultdict
import trimesh
import pymesh
import pickle
import sys
import volume_metrics as vm

ext_libs_path = os.path.join(os.path.dirname(__file__), 'scripts_for_gta')
sys.path.insert(0, ext_libs_path)
import volume_estimation as ve


def main(args):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    print("Input path:", args.input_path)
    print("Input type:", args.input_type)
    if args.input_type == "image":
        img_path_list = [args.input_path]
        base_dir = osp.dirname(osp.abspath(args.input_path))
        front_view_dir = side_view_dir = bbox_dir = base_dir
        result_filepath = f"{args.input_path[:-4]}_cliff_{args.backbone}.npz"
    else:
        if args.input_type == "video":
            basename = osp.basename(args.input_path).split('.')[0]
            base_dir = osp.join(osp.dirname(osp.abspath(args.input_path)), basename)
            img_dir = osp.join(base_dir, "imgs")
            front_view_dir = osp.join(base_dir, "front_view_%s" % args.backbone)
            side_view_dir = osp.join(base_dir, "side_view_%s" % args.backbone)
            bbox_dir = osp.join(base_dir, "bbox")
            result_filepath = osp.join(base_dir, f"{basename}_cliff_{args.backbone}.npz")
            if osp.exists(img_dir):
                print(f"Skip extracting images from video, because \"{img_dir}\" already exists")
            else:
                os.makedirs(img_dir, exist_ok=True)
                video_to_images(args.input_path, img_folder=img_dir)

        elif args.input_type == "folder":
            front_view_dir = osp.join(args.input_path, "front_view_%s" % args.backbone)
            side_view_dir = osp.join(args.input_path, "side_view_%s" % args.backbone)
            bbox_dir = osp.join(args.input_path, "bbox")
            basename = args.input_path.split('/')[-1]
            result_filepath = osp.join(args.input_path, f"{basename}_cliff_{args.backbone}.npz")
            
            if 'dataset' in args and args.dataset == 'agora':
                img_dir = args.input_path
                img_sub_dirs = glob.glob(os.path.join(img_dir, 'validation_*'))
                img_path_list = []
                imgs_list = []
                annotations_dict = dict()
                for img_sub_dir in img_sub_dirs:
                    sub_dir_idx = img_sub_dir.split('_')[-1]
                    imgs_in_subdir = glob.glob(os.path.join(img_sub_dir, f'val_{sub_dir_idx}', '*.png'))
                    img_path_list.extend(imgs_in_subdir)
                    imgs_list.extend([osp.basename(img_path) for img_path in imgs_in_subdir])
                
                    # read the GT for computing volume metrics
                    if args.compute_volume_metrics:
                        test_pkl_path = os.path.join(img_sub_dir, 'labels', f'validation_{sub_dir_idx}_withjv_vol.pkl')
                        with open(test_pkl_path, 'rb') as f:
                            test_pkl = pickle.load(f)
                            test_pkl = {k.split('.')[0]+'_1280x720.png':v for k,v in test_pkl.items()}
                        annotations_dict.update(test_pkl)
                    
                    if args.debug:
                        break
                        

            else:
                img_dir = osp.join(args.input_path, "imgs")

                # get all image paths
                img_path_list = glob.glob(osp.join(img_dir, '*.jpg'))
                img_path_list.extend(glob.glob(osp.join(img_dir, '*.png')))
                imgs_list = [osp.basename(img_path) for img_path in img_path_list]
                img_path_list.sort()

    # load all images
    print("Loading images ...")
    orig_img_bgr_all = [cv2.imread(img_path) for img_path in tqdm(img_path_list)]
    print("Image number:", len(img_path_list))
    
    if args.record_time:
        per_batch_det_times = []
        per_batch_est_times = []

    print("--------------------------- Detection ---------------------------")
    # Setup human detector
    human_detector = HumanDetector()
    det_batch_size = min(args.batch_size, len(orig_img_bgr_all))
    detection_dataset = DetectionDataset(orig_img_bgr_all, human_detector.in_dim)
    detection_data_loader = DataLoader(detection_dataset, batch_size=det_batch_size, num_workers=0)
    detection_all = []
    for batch_idx, batch in enumerate(tqdm(detection_data_loader)):
        if args.record_time:
            start_det_batch = torch.cuda.Event(enable_timing=True)
            end_det_batch = torch.cuda.Event(enable_timing=True)
            start_det_batch.record()
            
        norm_img = batch["norm_img"].to(device).float()
        dim = batch["dim"].to(device).float()

        detection_result = human_detector.detect_batch(norm_img, dim)
        detection_result[:, 0] += batch_idx * det_batch_size
        detection_all.extend(detection_result.cpu().numpy())
        
        if args.record_time:
            end_det_batch.record()
            # Waits for everything to finish running
            torch.cuda.synchronize()
            per_batch_det_times.append(start_det_batch.elapsed_time(end_det_batch))
            
    detection_all = np.array(detection_all)

    print("--------------------------- 3D HPS estimation ---------------------------")
    # Create the model instance
    cliff = eval("cliff_" + args.backbone)
    cliff_model = cliff(constants.SMPL_MEAN_PARAMS).to(device)
    # Load the pretrained model
    print("Load the CLIFF checkpoint from path:", args.ckpt)
    state_dict = torch.load(args.ckpt)['model']
    state_dict = strip_prefix_if_present(state_dict, prefix="module.")
    cliff_model.load_state_dict(state_dict, strict=True)
    cliff_model.eval()

    # Setup the SMPL model
    smpl_model = smplx.create(constants.SMPL_MODEL_DIR, "smpl").to(device)

    vol_pred_meshes_pymesh = defaultdict(list)
    vol_pred_meshes_bpy = defaultdict(list)
    pred_vert_arr = []
    if args.save_results:
        smpl_pose = []
        smpl_betas = []
        smpl_trans = []
        smpl_joints = []
        cam_focal_l = []

    mocap_db = MocapDataset(orig_img_bgr_all, detection_all, imgs_list)
    mocap_data_loader = DataLoader(mocap_db, batch_size=min(args.batch_size, len(detection_all)), num_workers=0)
    for batch in tqdm(mocap_data_loader):
        
        if args.record_time:
            start_est_batch = torch.cuda.Event(enable_timing=True)
            end_est_batch = torch.cuda.Event(enable_timing=True)
            start_est_batch.record()
        
        norm_img = batch["norm_img"].to(device).float()
        center = batch["center"].to(device).float()
        scale = batch["scale"].to(device).float()
        img_h = batch["img_h"].to(device).float()
        img_w = batch["img_w"].to(device).float()
        focal_length = batch["focal_length"].to(device).float()

        cx, cy, b = center[:, 0], center[:, 1], scale * 200
        bbox_info = torch.stack([cx - img_w / 2., cy - img_h / 2., b], dim=-1)
        # The constants below are used for normalization, and calculated from H36M data.
        # It should be fine if you use the plain Equation (5) in the paper.
        bbox_info[:, :2] = bbox_info[:, :2] / focal_length.unsqueeze(-1) * 2.8  # [-1, 1]
        bbox_info[:, 2] = (bbox_info[:, 2] - 0.24 * focal_length) / (0.06 * focal_length)  # [-1, 1]

        with torch.no_grad():
            pred_rotmat, pred_betas, pred_cam_crop = cliff_model(norm_img, bbox_info)

        # convert the camera parameters from the crop camera to the full camera
        full_img_shape = torch.stack((img_h, img_w), dim=-1)
        pred_cam_full = cam_crop2full(pred_cam_crop, center, scale, full_img_shape, focal_length)

        pred_output = smpl_model(betas=pred_betas,
                                 body_pose=pred_rotmat[:, 1:],
                                 global_orient=pred_rotmat[:, [0]],
                                 pose2rot=False,
                                 transl=pred_cam_full)
        pred_vertices = pred_output.vertices
        pred_vert_arr.extend(pred_vertices.cpu().numpy())
        
        if args.compute_volume_metrics:
            # Compute volume metrics
            for i in range(pred_vertices.shape[0]):
                img_id = batch["img_idx"][i]
                img_name = batch["img_name"][i]
                if args.pymesh:
                    pred_mesh_pymesh = pymesh.form_mesh(pred_vertices[i].cpu().numpy(), smpl_model.faces)
                    vol_pred_mesh = ve.compute_volume_of_selected_mesh_pymesh(pred_mesh_pymesh)
                    vol_pred_meshes_pymesh[img_name].append(vol_pred_mesh)
                if args.blender:
                    tmp_cache_obj_path = os.path.join('/media/odin/stdrr/data/.tmp_cache', f"pred_mesh_{img_id}.obj")
                    vertex_colors = np.ones([pred_vertices.shape[0], 4]) * [0.3, 0.3, 0.3, 0.8]
                    # process=False to avoid creating a new mesh
                    pred_mesh_trimesh = trimesh.Trimesh(pred_vertices[i].cpu().numpy(), smpl_model.faces, 
                                                        vertex_colors=vertex_colors, process=False)
                    pred_mesh_trimesh.export(tmp_cache_obj_path)
                    vol_pred_mesh = ve.compute_volume_from_obj_blender(tmp_cache_obj_path)
                    vol_pred_meshes_bpy[img_name].append(vol_pred_mesh)
        
        if args.record_time:
            end_est_batch.record()
            # Waits for everything to finish running
            torch.cuda.synchronize()
            per_batch_est_times.append(start_est_batch.elapsed_time(end_est_batch))
                    
        if args.save_results:
            if args.pose_format == "aa":
                rot_pad = torch.tensor([0, 0, 1], dtype=torch.float32, device=device).view(1, 3, 1)
                rot_pad = rot_pad.expand(pred_rotmat.shape[0] * 24, -1, -1)
                rotmat = torch.cat((pred_rotmat.view(-1, 3, 3), rot_pad), dim=-1)
                pred_pose = tgm.rotation_matrix_to_angle_axis(rotmat).contiguous().view(-1, 72)  # N*72
            else:
                pred_pose = pred_rotmat  # N*24*3*3

            smpl_pose.extend(pred_pose.cpu().numpy())
            smpl_betas.extend(pred_betas.cpu().numpy())
            smpl_trans.extend(pred_cam_full.cpu().numpy())
            smpl_joints.extend(pred_output.joints.cpu().numpy())
            cam_focal_l.extend(focal_length.cpu().numpy())
    
    if args.compute_volume_metrics:
        if args.record_time:
            per_batch_det_times = np.array(per_batch_det_times)
            per_batch_est_times = np.array(per_batch_est_times)
            per_batch_det_time = np.mean(per_batch_det_times)
            per_batch_est_time = np.mean(per_batch_est_times)
            instance_det_time = per_batch_det_time / det_batch_size
            instance_est_time = per_batch_est_time / det_batch_size
            total_batch_time = per_batch_det_time + per_batch_est_time
            total_instance_time = instance_det_time + instance_est_time
        else:
            instance_det_time = -1.
            instance_est_time = -1.
            total_batch_time = -1.
            total_instance_time = -1.
        gt_vols, gt_counts = [], []
        pred_vols_pymesh = []
        pred_vols_bpy = []
        for img_name in annotations_dict:
            frame_gt_vol = np.sum(annotations_dict[img_name]['volumes'])
            frame_gt_count = len(annotations_dict[img_name]['volumes'])
            gt_counts.append(frame_gt_count)
            gt_vols.append(frame_gt_vol)
            if args.pymesh:
                frame_pred_vol_pymesh = np.sum(vol_pred_meshes_pymesh[img_name])
                pred_vols_pymesh.append(frame_pred_vol_pymesh)
            if args.blender:
                frame_pred_vol_bpy = np.sum(vol_pred_meshes_bpy[img_name])
                pred_vols_bpy.append(frame_pred_vol_bpy)
                
        gt_vols = torch.tensor(gt_vols)
        gt_counts = torch.tensor(gt_counts)
        with open('volume_result.csv', 'w') as f:
            f.write('lib,mae,mse,nae,ppmae,b_det_t,i_det_t,b_est_t,i_est_t,b_tot_t,i_tot_t\n')
            if args.pymesh:
                pred_vols_pymesh = torch.tensor(pred_vols_pymesh)
                mae = vm.MAE(pred_vols_pymesh, gt_vols)
                mse = vm.MSE(pred_vols_pymesh, gt_vols)
                nae = vm.NAE(pred_vols_pymesh, gt_vols)
                ppmae = vm.PPMAE(pred_vols_pymesh, gt_vols, gt_counts)
                print('---------- Pymesh -----------')
                print(f"MAE: {mae}")
                print(f"MSE: {mse}")
                print(f"NAE: {nae}")
                print(f"PPMAE: {ppmae}")
                f.write(f'pymesh,{mae},{mse},{nae},{ppmae},{per_batch_det_time},{instance_det_time},{per_batch_est_time},{instance_est_time},{total_batch_time},{total_instance_time}\n')
                
            if args.blender:
                pred_vols_bpy = torch.tensor(pred_vols_bpy)
                mae = vm.MAE(pred_vols_bpy, gt_vols)
                mse = vm.MSE(pred_vols_bpy, gt_vols)
                nae = vm.NAE(pred_vols_bpy, gt_vols)
                ppmae = vm.PPMAE(pred_vols_bpy, gt_vols, gt_counts)
                print('---------- Bpy -----------')
                print(f"MAE: {mae}")
                print(f"MSE: {mse}")
                print(f"NAE: {nae}")
                print(f"PPMAE: {ppmae}")
                f.write(f'bpy,{mae},{mse},{nae},{ppmae},{per_batch_det_time},{instance_det_time},{per_batch_est_time},{instance_est_time},{total_batch_time},{total_instance_time}\n')

    if args.save_results:
        print(f"Save results to \"{result_filepath}\"")
        np.savez(result_filepath, imgname=img_path_list,
                 pose=smpl_pose, shape=smpl_betas, global_t=smpl_trans,
                 pred_joints=smpl_joints, focal_l=cam_focal_l,
                 detection_all=detection_all)
        np.savez(result_filepath, imgname=img_path_list, pred_vol=vol_pred_meshes_pymesh, gt_vol=gt_vols)

    print("--------------------------- Visualization ---------------------------")
    # make the output directory
    os.makedirs(front_view_dir, exist_ok=True)
    print("Front view directory:", front_view_dir)
    if args.show_sideView:
        os.makedirs(side_view_dir, exist_ok=True)
        print("Side view directory:", side_view_dir)
    if args.show_bbox:
        os.makedirs(bbox_dir, exist_ok=True)
        print("Bounding box directory:", bbox_dir)

    pred_vert_arr = np.array(pred_vert_arr)
    for img_idx, orig_img_bgr in enumerate(tqdm(orig_img_bgr_all)):
        chosen_mask = detection_all[:, 0] == img_idx
        chosen_vert_arr = pred_vert_arr[chosen_mask]

        # setup renderer for visualization
        img_h, img_w, _ = orig_img_bgr.shape
        focal_length = estimate_focal_length(img_h, img_w)
        renderer = Renderer(focal_length=focal_length, img_w=img_w, img_h=img_h,
                            faces=smpl_model.faces,
                            same_mesh_color=(args.input_type == "video"))
        front_view = renderer.render_front_view(chosen_vert_arr,
                                                bg_img_rgb=orig_img_bgr[:, :, ::-1].copy())

        # save rendering results
        basename = osp.basename(img_path_list[img_idx]).split(".")[0]
        filename = basename + "_front_view_cliff_%s.jpg" % args.backbone
        front_view_path = osp.join(front_view_dir, filename)
        cv2.imwrite(front_view_path, front_view[:, :, ::-1])

        if args.show_sideView:
            side_view_img = renderer.render_side_view(chosen_vert_arr)
            filename = basename + "_side_view_cliff_%s.jpg" % args.backbone
            side_view_path = osp.join(side_view_dir, filename)
            cv2.imwrite(side_view_path, side_view_img[:, :, ::-1])

        # delete the renderer for preparing a new one
        renderer.delete()

        # draw the detection bounding boxes
        if args.show_bbox:
            chosen_detection = detection_all[chosen_mask]
            bbox_info = chosen_detection[:, 1:6]

            bbox_img_bgr = orig_img_bgr.copy()
            for min_x, min_y, max_x, max_y, conf in bbox_info:
                ul = (int(min_x), int(min_y))
                br = (int(max_x), int(max_y))
                cv2.rectangle(bbox_img_bgr, ul, br, color=(0, 255, 0), thickness=2)
                cv2.putText(bbox_img_bgr, "%.1f" % conf, ul,
                            cv2.FONT_HERSHEY_COMPLEX_SMALL, fontScale=1.0, color=(0, 0, 255), thickness=1)
            filename = basename + "_bbox.jpg"
            bbox_path = osp.join(bbox_dir, filename)
            cv2.imwrite(bbox_path, bbox_img_bgr)

    # make videos
    if args.make_video:
        print("--------------------------- Making videos ---------------------------")
        from common.utils import images_to_video
        images_to_video(front_view_dir, video_path=front_view_dir + ".mp4", frame_rate=args.frame_rate)
        if args.show_sideView:
            images_to_video(side_view_dir, video_path=side_view_dir + ".mp4", frame_rate=args.frame_rate)
        if args.show_bbox:
            images_to_video(bbox_dir, video_path=bbox_dir + ".mp4", frame_rate=args.frame_rate)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--input_type', default='image', choices=['image', 'folder', 'video'],
                        help='input type')
    parser.add_argument('--input_path', default='test_samples/nba.jpg', help='path to the input data')

    parser.add_argument('--ckpt',
                        default="data/ckpt/hr48-PA43.0_MJE69.0_MVE81.2_3dpw.pt",
                        help='path to the pretrained checkpoint')
    parser.add_argument("--backbone", default="hr48", choices=['res50', 'hr48'],
                        help="the backbone architecture")
    parser.add_argument('--batch_size', type=int, default=32,
                        help='batch size for detection and motion capture')

    parser.add_argument('--save_results', action='store_true',
                        help='save the results as a npz file')
    parser.add_argument('--pose_format', default='aa', choices=['aa', 'rotmat'],
                        help='aa for axis angle, rotmat for rotation matrix')

    parser.add_argument('--show_bbox', action='store_true',
                        help='show the detection bounding boxes')
    parser.add_argument('--show_sideView', action='store_true',
                        help='show the result from the side view')

    parser.add_argument('--make_video', action='store_true',
                        help='make a video of the rendering results')
    parser.add_argument('--frame_rate', type=int, default=30, help='frame rate')
    parser.add_argument('--debug', action='store_true', default=False)
    parser.add_argument('--dataset', default='agora', help='the dataset for evaluation')
    parser.add_argument('--compute_volume_metrics', action='store_true', help='compute volume metrics')
    parser.add_argument('--record_time', action='store_true', help='record time')
    # Additional option for computing the volume
    sub_parser = parser.add_subparsers(dest='vol_parser')
    vol_parser = sub_parser.add_parser('vol_parser')
    vol_parser.add_argument('--pymesh', action='store_true', default=False)
    vol_parser.add_argument('--blender', action='store_true', default=False)

    args = parser.parse_args()
    main(args)
