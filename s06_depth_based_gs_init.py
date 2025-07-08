'''
    Steps:
        1. Keep the depth x_1 -eg. depth_1_1.npy, (x is the round index of training)
        2. supplement the depth using depth_1_2.npy 
            2.1 keep the depth within the range of the depth_1_1.npy

'''
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import cv2
import torch
import argparse
import logging
import pprint
import random
import warnings
import numpy as np
import open3d as o3d
import torch.backends.cudnn as cudnn
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.optim import AdamW
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter


from dataset.wildtrack import WildtrackDepthEstimation, SuperPixels
from submodules.depth_anything_v2.dpt import DepthAnythingV2
from submodules.depth_anything_v2.metric_depth.util.loss import SiLogLoss
from submodules.depth_anything_v2.metric_depth.util.metric import eval_depth
from submodules.depth_anything_v2.metric_depth.util.utils import init_log
from ProteusLib import depth_constraints
from s04_gs_us import bg_filter_fn, depth_filter_fn

def depth2points(depth, K, c2w):
    height, width = depth.shape
    u, v = np.meshgrid(np.arange(width), np.arange(height))
    x = (u - K[0, 2]) * depth / K[0, 0]
    y = (v - K[1, 2]) * depth / K[1, 1]
    z = depth
    points = np.stack([x.flatten(), y.flatten(), z.flatten()], axis=-1)
    points = c2w[:3, :3] @ points.T + c2w[:3, 3:4]
    return points.T.reshape(height, width, 3)


def cylindermask(human_mask, points1, depth_mask1, points2=None, depth_mask2=None, num_sample=10, clip_len=80, sample_type='uniform', cam_idx=None):
    new_points = []
    for human_id in np.unique(human_mask):
        if human_id == 0:
            continue
        mask = human_mask == human_id
        points_ = points1[mask]
        depth_mask_ = depth_mask1[mask]
        points_ = points_[depth_mask_]
        if points_.shape[0] == 0 and points2 is not None and depth_mask2 is not None:
            points_ = points2[mask]
            depth_mask_ = depth_mask2[mask]
            points_ = points_[depth_mask_]
        if points_.shape[0] == 0:
            continue
        x_min, x_max = points_[:, 0].min(), points_[:, 0].max()
        y_min, y_max = points_[:, 1].min(), points_[:, 1].max()
        z_min, z_max = points_[:, 2].min(), points_[:, 2].max()
        z_min = min(z_min, 0)
        x_median = np.median(points_[:, 0])
        y_median = np.median(points_[:, 1])
        
        x_len = min((x_max - x_min), clip_len)
        x_min = x_median - x_len / 2
        x_max = x_median + x_len / 2
        y_len = min((y_max - y_min), clip_len)
        y_min = y_median - y_len / 2
        y_max = y_median + y_len / 2
        
        # uniform sampling within the bounding box
        if sample_type == 'uniform':
            x = np.linspace(x_min, x_max, num_sample)
            y = np.linspace(y_min, y_max, num_sample)
            z = np.linspace(z_min, z_max, num_sample*5)
            xyz = np.stack(np.meshgrid(x, y, z, indexing='ij'), axis=-1)
            new_points.append(xyz.reshape(-1, 3))
        else:
            x = np.random.uniform(x_min, x_max, num_sample)
            y = np.random.uniform(y_min, y_max, num_sample)
            z = np.random.uniform(z_min, z_max, num_sample)
            new_points.append(np.stack([x, y, z], axis=-1))
    return np.concatenate(new_points, axis=0)

def cylindermask_based_init():
    args_src = ['--root', '/home/jiahao/Downloads/data/wildtrack_data', 
                '--round', '1_1']
    parser = argparse.ArgumentParser(description='Depth based gaussian initlization')
    parser.add_argument("--root", type=str, required=True, help="path to image folder")
    parser.add_argument("--round", type=str, required=True, help="round name")
    args = parser.parse_args(args_src)
    data_root = args.root
    round_idx = args.round.split('_')[0]
    next_round_idx = int(round_idx) + 1
    depth_root = os.path.join(data_root, 'depths')

    ori_size = (540, 960)
    dataset_os = WildtrackDepthEstimation(data_root, mode='val', size=ori_size, depth_mask_type='sam_label', mask_choice='foreground')
    w2cs = dataset_os.w2cs
    Ks = dataset_os.intrinsic_matrices
    
    depth_folders = [ os.path.join(depth_root, p) for p in sorted(os.listdir(depth_root)) if p != 'background'] 
    for frame_idx in range(len(depth_folders)):
        print(f'Processing frame: {frame_idx:08d}')
        save_root = depth_folders[frame_idx]
        split_mask = dataset_os[frame_idx]['split_mask']
        
        depths1 = np.load(os.path.join(save_root, f'depth_{round_idx}_1.npy'))
        depths2 = np.load(os.path.join(save_root, f'depth_{round_idx}_2.npy'))
        depth_masks1 = np.load(os.path.join(save_root, f'mask_{round_idx}_1.npy'))
        depth_masks2 = np.load(os.path.join(save_root, f'mask_{round_idx}_2.npy'))
        fused_points = []
        # gs_points = []
        # depth_prior_points = []
        depth_max_ratio=1.0
        for cam_idx in range(dataset_os.num_cam):
            
            # print('Fusing cam', cam_idx)
            depth_mask1 = depth_masks1[cam_idx].astype(np.bool_)
            depth_mask2 = depth_masks2[cam_idx].astype(np.bool_)
            
            points1 = depth2points(depths1[cam_idx], Ks[cam_idx], np.linalg.inv(w2cs[cam_idx]))
            points2 = depth2points(depths2[cam_idx], Ks[cam_idx], np.linalg.inv(w2cs[cam_idx]))
            
            depth_mask1 = depth_mask1 & (depths1[cam_idx] < depths1[cam_idx].max() * depth_max_ratio)
            depth_mask2 = depth_mask2 & (depths2[cam_idx] < depths2[cam_idx].max() * depth_max_ratio)
            # gs_points.append(points1.reshape(-1, 3)[depth_mask1.reshape(-1)])
            # depth_prior_points.append(points2.reshape(-1, 3)[depth_mask2.reshape(-1)])
            
            # fused_points.append(cylindermask(split_mask[cam_idx], points1, depth_mask1, points2, depth_mask2, sample_type='uniform', cam_idx=cam_idx))
            fused_points.append(cylindermask(split_mask[cam_idx], points2, depth_mask2, points2, depth_mask2, sample_type='uniform', cam_idx=cam_idx))
            fused_points.append(cylindermask(split_mask[cam_idx], points1, depth_mask1, points1, depth_mask1, sample_type='uniform', cam_idx=cam_idx))
            
            
        # gs_points = np.concatenate(gs_points, axis=0)
        # depth_prior_points = np.concatenate(depth_prior_points, axis=0)
        
        fused_points = np.concatenate(fused_points, axis=0)
        pcd_gs_vx = o3d.geometry.PointCloud()
        pcd_gs_vx.points = o3d.utility.Vector3dVector(fused_points)
        # pcd_gs_vx.paint_uniform_color([0, 0, 0])
        # o3d.visualization.draw_geometries([pcd_gs_vx])
        
        o3d.io.write_point_cloud(os.path.join(save_root, f'init_gs_{next_round_idx}_1.ply'), pcd_gs_vx)
    
def depth_based_init(bg_filter=False):
    args_src = ['--root', '/home/jiahao/Downloads/data/wildtrack_data', 
                '--round', '2_1']
    parser = argparse.ArgumentParser(description='Depth based gaussian initlization')
    parser.add_argument("--root", type=str, required=True, help="path to image folder")
    parser.add_argument("--round", type=str, required=True, help="round name")
    parser.add_argument("--human_depth_intvl", type=float, default=40.0, help="the depth threshold for human")
    parser.add_argument("--depth_supp_num", type=int, default=200, help="the supplement number of depth samples for each superpixel")
    args = parser.parse_args(args_src) # FOR DEBUG
    # args = parser.parse_args()
    data_root = args.root
    round_idx = args.round.split('_')[0]
    next_round_idx = int(round_idx) + 1
    depth_root = os.path.join(data_root, 'depths')

    ori_size = (540, 960)
    H, W = ori_size
    dataset_os = WildtrackDepthEstimation(data_root, mode='val', size=ori_size, depth_mask_type='sam_label', mask_choice='foreground')
    w2cs = dataset_os.w2cs
    Ks = dataset_os.intrinsic_matrices
    c2ws = [np.linalg.inv(w2c) for w2c in w2cs]
    sp_dataset = SuperPixels(data_root, n_segments=30, HW=dataset_os.image_feat_size)
    
    unit_scale = 100 # cm to m
    num_depth_bin = 2_000
    max_depth = 40.0
    rel_depth = np.linspace(0.05, 1, num_depth_bin)
    rel_depth = rel_depth * unit_scale * max_depth
    
    human_depth_intvl = np.linspace(- 0.5 * args.human_depth_intvl, 0.5 * args.human_depth_intvl, args.depth_supp_num)
    
    depth_folders = [ os.path.join(depth_root, p) for p in sorted(os.listdir(depth_root)) if p != 'background'] 
    for frame_idx in range(len(depth_folders)):
        print(f'Processing frame: {frame_idx:08d}')
        save_root = depth_folders[frame_idx]
        split_mask = dataset_os[frame_idx]['split_mask']
        
        depths1 = np.load(os.path.join(save_root, f'depth_{round_idx}_1.npy'))
        depths2 = np.load(os.path.join(save_root, f'depth_{round_idx}_2.npy'))
        depth_masks1 = np.load(os.path.join(save_root, f'mask_{round_idx}_1.npy'))
        depth_masks2 = np.load(os.path.join(save_root, f'mask_{round_idx}_2.npy'))
        fused_points = []
        
        super_pixels = sp_dataset[frame_idx]
        for cam_idx in range(dataset_os.num_cam):
            sps_image = super_pixels['image'][cam_idx]
            sps_perframe = super_pixels['centroids'][cam_idx]
            sps_idx = super_pixels['centroids_idx'][cam_idx]
            for human_idx in np.unique(sps_idx):
                sps = sps_perframe[human_idx == sps_idx].astype(np.int32)
                valid_mask = depth_masks2[cam_idx][sps[:, 0], sps[:, 1]] #cnts[:, 0].astype(np.int32), cnts[:, 1].astype(np.int32)
                spltm = split_mask[cam_idx]
                m_id, counts = np.unique(spltm[sps[:, 0], sps[:, 1]], return_counts=True)
                m_id = m_id[counts.argmax()]
                if np.all(valid_mask == False):
                    # case1: all superpixels fall in empty mask
                    valid_mask = depth_masks1[cam_idx][sps[:, 0], sps[:, 1]] 
                    if np.all(valid_mask == False):
                        continue
                    valid_depth_mask = (m_id == spltm) & depth_masks1[cam_idx]
                    human_depths = depths1[cam_idx][valid_depth_mask]
                    valid_human_depth = depths1[cam_idx][sps[valid_mask, 0], sps[valid_mask, 1]]
                    
                else:
                    # case2: part of superpixels fall in empty mask, find the median depth of the superpixels
                    # case3: all superpixels fall in human mask, calculate the depth range of the superpixels
                    valid_depth_mask = (m_id == spltm) & depth_masks2[cam_idx]
                    human_depths = depths2[cam_idx][valid_depth_mask]
                    valid_human_depth = depths2[cam_idx][sps[valid_mask, 0], sps[valid_mask, 1]]
                    
                valid_human_rgb = sps_image[sps[:, 0], sps[:, 1]]
                valid_human_rgb = np.repeat(valid_human_rgb[:, None, :], repeats=human_depth_intvl.shape[0], axis=1)
                median_depth = np.median(human_depths)    
                valid_human_depth = valid_human_depth.reshape(-1, 1) + human_depth_intvl.reshape(1, -1)
                x = (sps[valid_mask, 1] - Ks[cam_idx][0, 2]) / Ks[cam_idx][0, 0]
                y = (sps[valid_mask, 0] - Ks[cam_idx][1, 2]) / Ks[cam_idx][1, 1]
                x = x.reshape(-1, 1) * valid_human_depth
                y = y.reshape(-1, 1) * valid_human_depth
                z = valid_human_depth
                pts3d = np.stack([x, y, z], axis=-1).reshape(-1, 3)
                pts3d_world_xyz = c2ws[cam_idx][:3, :3] @ pts3d.T + c2ws[cam_idx][:3, 3:4]
                pts3d_world_rgb = valid_human_rgb[valid_mask].reshape(-1, 3)
                pts3d_world = np.concatenate([pts3d_world_xyz.T, pts3d_world_rgb], axis=1)
                fused_points.append(pts3d_world)
                
                x = (sps[~valid_mask, 1] - Ks[cam_idx][0, 2]) / Ks[cam_idx][0, 0]
                y = (sps[~valid_mask, 0] - Ks[cam_idx][1, 2]) / Ks[cam_idx][1, 1]
                median_depth = np.repeat(median_depth, repeats=x.shape[0])
                median_depth = median_depth.reshape(-1, 1) + human_depth_intvl.reshape(1, -1)
                x = x.reshape(-1, 1) * median_depth
                y = y.reshape(-1, 1) * median_depth
                z = median_depth
                pts3d = np.stack([x, y, z], axis=-1).reshape(-1, 3)
                pts3d_world_xyz = c2ws[cam_idx][:3, :3] @ pts3d.T + c2ws[cam_idx][:3, 3:4]
                pts3d_world_rgb = valid_human_rgb[~valid_mask].reshape(-1, 3)
                pts3d_world = np.concatenate([pts3d_world_xyz.T, pts3d_world_rgb], axis=1)
                fused_points.append(pts3d_world)
                    
        fused_points = np.concatenate(fused_points, axis=0)
        if bg_filter:
            pts3d_index = np.arange(fused_points.shape[0])
            pts3d_index_del = []
            
            comb_masks = dataset_os[frame_idx]['valid_mask']
            for cam_idz in range(dataset_os.num_cam):
                pts3d_index_del.append(bg_filter_fn(w2c=w2cs[cam_idz],
                            K=Ks[cam_idz],
                            pts3d_world=fused_points[:, :3].T,
                            pts3d_index=pts3d_index, 
                            masks_comb=comb_masks[cam_idz].astype(np.bool_),
                            H=H, W=W))
        
            pts3d_index_del = np.concatenate(pts3d_index_del, axis=0)
            pts3d_index_del = np.unique(pts3d_index_del)
            pts3d_index_keep1 = np.setdiff1d(pts3d_index, pts3d_index_del)
            fused_points = fused_points[pts3d_index_keep1]
        
        nan_mask = np.isnan(fused_points).any(axis=1)
        fused_points = fused_points[~nan_mask]
        
        pcd_gs_vx = o3d.geometry.PointCloud()
        pcd_gs_vx.points = o3d.utility.Vector3dVector(fused_points[:, :3])
        # visualize the fused points
        pcd_gs_vx.colors = o3d.utility.Vector3dVector(fused_points[:, 3:])
        # o3d.visualization.draw_geometries([pcd_gs_vx])
        o3d.io.write_point_cloud(os.path.join(save_root, f'init_gs_{next_round_idx}_1.ply'), pcd_gs_vx)
        print(os.path.join(save_root, f'init_gs_{next_round_idx}_1.ply'), ' has been saved!')
                    
                
        
if __name__ == '__main__':
    # cylindermask_based_init()
    depth_based_init(bg_filter=True)