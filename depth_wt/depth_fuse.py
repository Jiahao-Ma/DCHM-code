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
wildtrack = WildtrackDepthEstimation("/home/jiahao/Downloads/data/wildtrack_data",
                    mode='val', size=[540, 960], depth_mask_type='matching_label', mask_choice='background')
data = wildtrack[-1]
novel_view = wildtrack.novel_view

depth_type = 'rel_depth'
if depth_type == 'rel_depth':
    depth_model = pipeline(task="depth-estimation", model="depth-anything/Depth-Anything-V2-Large-hf", device=0)
elif depth_type == 'metric_depth':
    metric_depth_model = get_model(encoder='vitl')

all_pcs = []
depth_scale = 1
threshold = 3
# scales = np.load('depth_wt/scales.npy')
shifts_scales = np.load('depth_wt/shifts_scales.npy', allow_pickle=True).item()
shifts = shifts_scales['shifts']
scales = shifts_scales['scales']

for cam_idx in range(1, 8):
    gt_depth = data['depth'][cam_idx-1]
    gt_mask = data['valid_mask'][cam_idx-1].astype(bool)
    h, w = gt_depth.shape
    # image = Image.open(f'data/00000001/cam{cam_idx}.png')
    image = Image.open(f'/home/jiahao/Downloads/data/wildtrack_data/Image_subsets/{data["frame_idx"]}/cam{cam_idx}.png')
    if depth_type == 'rel_depth':
        results = depth_model(image)
        # resize image to the same size as the depth image
        rgb = np.array(image.resize((w, h)))#.reshape(-1, 3)
        depth_img = results["depth"]
        disparity = results['predicted_depth']
        disparity = F.interpolate(disparity[:, None], (h, w), mode="bilinear", align_corners=True)[0, 0]
        disparity_norm = (disparity - disparity.min()) / (disparity.max() - disparity.min())
        depth_norm, mask = DisparityToDepth()(torch.tensor(disparity_norm), depth_min=0.01, depth_max=5)
        depth = 1 / disparity
        # mask = mask.cpu().numpy().astype(bool)
        # mask = (depth > depth.min()*1.2) & (depth < depth.max()*0.5)
        depth = depth.cpu().numpy()
    elif depth_type == 'metric_depth':
        imageNP = np.array(image)
        rgb = np.array(image.resize((w, h)))
        depth = metric_depth_model.infer_image(imageNP)
        depth = cv2.resize(depth, (w, h), interpolation=cv2.INTER_NEAREST)
        mask = None
    
    depth_grad_x = np.gradient(depth_norm, axis=0)
    depth_grad_y = np.gradient(depth_norm, axis=1)
    depth_grad = np.stack((depth_grad_x, depth_grad_y), axis=-1)
    depth_grad = np.linalg.norm(depth_grad, axis=-1)
    region_with_valid_depth_gradient = depth_grad <= np.median(depth_grad*threshold)
    if mask is not None:
        mask = np.logical_and(mask, region_with_valid_depth_gradient)
    else:
        mask = region_with_valid_depth_gradient
    mask = mask.cpu().numpy().astype(np.bool_)
    z = depth * scales[cam_idx-1] + shifts[cam_idx-1]
    # z = np.where(mask, z, np.zeros_like(z))
    # z *= scales[cam_idx-1]
    
    xx, yy = np.meshgrid(np.arange(0, depth.shape[1]), np.arange(0, depth.shape[0]))
    x = ( xx - wildtrack.intrinsic_matrices[cam_idx-1][0, 2] ) * z / wildtrack.intrinsic_matrices[cam_idx-1][0, 0]
    y = ( yy - wildtrack.intrinsic_matrices[cam_idx-1][1, 2] ) * z / wildtrack.intrinsic_matrices[cam_idx-1][1, 1]
    # contrust point cloud
    point_cloud = np.stack((x, y, z), axis=-1)
    point_cloud = point_cloud[mask]
    point_cloud = point_cloud.reshape(-1, 3)
    rgb = rgb[mask].reshape(-1, 3)
    c2w = novel_view[cam_idx-1]['c2w']
    point_cloud = c2w[:3, :3] @ point_cloud.T + c2w[:3, 3:]
    point_cloud_w_color = np.concatenate([point_cloud.T, rgb], axis=1)
    all_pcs.append(point_cloud_w_color)
    
for i, pc in enumerate(all_pcs):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pc[:, :3])
    pcd.colors = o3d.utility.Vector3dVector(pc[:, 3:] / 255)

    # pcd = pcd.voxel_down_sample(voxel_size=1)
    # visualize
    # o3d.visualization.draw_geometries([pcd])
    o3d.io.write_point_cloud(f"depth_wt/fuse_pc_{i}.ply", pcd)
    
all_pcs  = np.concatenate(all_pcs, axis=0)
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(all_pcs[:, :3])
pcd.colors = o3d.utility.Vector3dVector(all_pcs[:, 3:] / 255)

pcd = pcd.voxel_down_sample(voxel_size=1)
# visualize
# o3d.visualization.draw_geometries([pcd])
o3d.io.write_point_cloud("depth_wt/fuse_pc.ply", pcd)

