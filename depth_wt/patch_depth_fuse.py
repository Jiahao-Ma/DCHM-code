import os, sys;sys.path.append(os.getcwd())
import cv2
import torch    
import open3d as o3d
from transformers import pipeline
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from ProteusLib import DisparityToDepth
from dataset.wildtrack import WildtrackDepthEstimation
import torch.nn.functional as F
from depth_wt.cal_scale import get_model

def depth2points(depth, rgb, K, c2w, h, w):
    y, x = np.meshgrid(np.arange(h), np.arange(w))
    points = np.stack([x, y, np.ones_like(x)], axis=-1).reshape(-1, 3)
    points = points * depth.reshape(-1, 1)
    points = np.linalg.inv(K) @ points.T
    points = c2w[:3, :3] @ points + c2w[:3, 3:]
    mask = depth.reshape(-1) != 0
    rgb = rgb.reshape(-1, 3)[mask]
    points = points.T[mask]
    points = np.concatenate([points, rgb], axis=-1)  
    return points

def load_shifts_and_scales(file_path):
    data = np.load(file_path, allow_pickle=True).item()
    return data['shifts'], data['scales'], data['depths'], data['masks']

wildtrack = WildtrackDepthEstimation("/home/jiahao/Downloads/data/wildtrack_data",
                    mode='val', size=[540, 960], depth_mask_type='matching_label', mask_choice='background')
data = wildtrack[0]
novel_view = wildtrack.novel_view

# depth_type = 'rel_depth'
# if depth_type == 'rel_depth':
#     depth_model = pipeline(task="depth-estimation", model="depth-anything/Depth-Anything-V2-Large-hf", device=0)
# elif depth_type == 'metric_depth':
#     metric_depth_model = get_model(encoder='vitl')

all_pcs = []
depth_scale = 1
threshold = 3
shifts, scales, depths, masks = load_shifts_and_scales('depth_wt/00000000_scales_shifts.npy')

for cam_idx in range(7):
    gt_depth = data['depth'][cam_idx]
    gt_mask = data['valid_mask'][cam_idx].astype(bool)
    h, w = gt_depth.shape
    # image = Image.open(f'data/00000001/cam{cam_idx}.png')
    image = Image.open(f'/home/jiahao/Downloads/data/wildtrack_data/Image_subsets/00000000/cam{cam_idx+1}.png')
    shift = shifts[cam_idx]
    scale = scales[cam_idx]
    rel_depth = depths[cam_idx]
    mask = masks[cam_idx]
    h, w = rel_depth.shape
    rgb = np.array(image.resize((w, h)))
    metric_depth = rel_depth * scale + shift
    depth_grad_x = np.gradient(metric_depth, axis=0)
    depth_grad_y = np.gradient(metric_depth, axis=1)
    depth_grad = np.stack((depth_grad_x, depth_grad_y), axis=-1)
    depth_grad = np.linalg.norm(depth_grad, axis=-1)
    region_with_valid_depth_gradient = depth_grad <= np.median(depth_grad*threshold)
    if mask is not None:
        mask = np.logical_and(mask, region_with_valid_depth_gradient)
    else:
        mask = region_with_valid_depth_gradient
    xx, yy = np.meshgrid(np.arange(0, metric_depth.shape[1]), np.arange(0, metric_depth.shape[0]))
    x = ( xx - wildtrack.intrinsic_matrices[cam_idx-1][0, 2] ) * metric_depth / wildtrack.intrinsic_matrices[cam_idx-1][0, 0]
    y = ( yy - wildtrack.intrinsic_matrices[cam_idx-1][1, 2] ) * metric_depth / wildtrack.intrinsic_matrices[cam_idx-1][1, 1]
    # contrust point cloud
    point_cloud = np.stack((x, y, metric_depth), axis=-1)
    point_cloud = point_cloud[mask]
    point_cloud = point_cloud.reshape(-1, 3)
    rgb = rgb[mask].reshape(-1, 3)
    c2w = novel_view[cam_idx-1]['c2w']
    point_cloud = c2w[:3, :3] @ point_cloud.T + c2w[:3, 3:]
    point_cloud_w_color = np.concatenate([point_cloud.T, rgb], axis=1)
    all_pcs.append(point_cloud_w_color)
    

all_pcs  = np.concatenate(all_pcs, axis=0)
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(all_pcs[:, :3])
pcd.colors = o3d.utility.Vector3dVector(all_pcs[:, 3:] / 255)

# pcd = pcd.voxel_down_sample(voxel_size=1)
# visualize
# o3d.visualization.draw_geometries([pcd])
o3d.io.write_point_cloud("depth_wt/fuse_patch_pc.ply", pcd)
