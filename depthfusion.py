import cv2, os
import numpy as np
import matplotlib.pyplot as plt
from dataset.wildtrack import Wildtrack
import torch
import open3d as o3d
from s04_gs_us import bg_filter_fn
from skimage import exposure
def depth2points_fun(depth, mask, K):
    v, u = np.where(mask)
    # u, v = np.meshgrid(np.arange(W), np.arange(H))
    u = u.flatten()
    v = v.flatten()
    z = depth[mask]
    x = (u - K[0, 2]) * z / K[0, 0]
    y = (v - K[1, 2]) * z / K[1, 1]
    points = np.stack([x, y, z], axis=1)
    return points
def depth_filter(depth, ratio=0.5, max_depth=30):
    if isinstance(depth, np.ndarray):
        depth = torch.tensor(depth)
    grad_x = torch.abs(depth[:, 1:] - depth[:, :-1])  # Horizontal gradient
    grad_y = torch.abs(depth[1:, :] - depth[:-1, :])  # Vertical gradient

    # Pad the gradients to maintain the same size as the input depth map
    grad_x = torch.nn.functional.pad(grad_x, (0, 1), mode='constant', value=0)  # Pad last column
    grad_y = torch.nn.functional.pad(grad_y, (0, 0, 0, 1), mode='constant', value=0)  # Pad last row

    # Combine the gradients (you can also use the maximum or another metric)
    grad_magnitude = grad_x + grad_y

    # Create the mask: True for regions with gradient < threshold, False otherwise
    if ratio is not None:
        threshold = grad_magnitude.mean() * ratio
        mask = grad_magnitude < threshold
    else:
        mask = torch.ones_like(grad_magnitude, dtype=torch.bool)
        
    if max_depth is not None:
        mask = mask & (depth < max_depth) & (depth > 0)
    else:
        mask = mask & (depth > 0)

    return mask.cpu().numpy()


depth_propath = np.load(r'submodules/depthpro/data/wildtrack/00000000/pred_depths.npy')

depth_anything = np.load(r'submodules/depthpro/data/wildtrack/00000000/pred_depths_depthanything.npy')
# plt.imshow(depth_propath[0])
# plt.show()
num_cam=7
foreground_depths = np.load('/home/jiahao/Downloads/data/wildtrack_data_gt/depths/00000000/gs_optimize/depth_1_2.npy')
foreground_masks = np.load('/home/jiahao/Downloads/data/wildtrack_data_gt/depths/00000000/gs_optimize/mask_1_2.npy')
background_depths = np.load('/home/jiahao/Downloads/data/wildtrack_data/depths/background/depths.npy')
image_paths = ['/home/jiahao/Downloads/data/wildtrack_data_gt/Image_subsets/00000000/cam{}.png'.format(camId) for camId in range(1,8)]
predefined_background_depth_masks = []
for cam_idx in range(num_cam):
    depth = background_depths[cam_idx]
    depth_mask = np.zeros_like(depth, dtype=np.bool_)
    depth_mask[depth > 0] = True
    predefined_background_depth_masks.append(depth_mask)
background_depth_mask_paths = [f'/home/jiahao/Downloads/data/wildtrack_data_gt/masks/ground/00000000/cam{camId}.npy' for camId in range(1,8)]

width, height = 960, 540
dataset = Wildtrack('/home/jiahao/Downloads/data/wildtrack_data_gt',mask_type='people', mask_label_type='split')
data = dataset[0]
fuse_foreground_pcs = []
fuse_background_pcs = []
for camId in range(num_cam):
    foreground_depth = foreground_depths[camId]
    foreground_mask = foreground_masks[camId]
    print('Processing camera {}'.format(camId))
    if camId == 2:
        depth_prior = cv2.resize(depth_anything[camId], (width, height), interpolation=cv2.INTER_NEAREST)
    else:
        depth_prior = cv2.resize(depth_propath[camId], (width, height), interpolation=cv2.INTER_NEAREST)
    humanMasks = data['mask'][camId]
    # final_foreground_depth = foreground_depth
    # final_foreground_mask = foreground_mask
    # plt.figure(figsize=(20, 10))
    # plt.subplot(131)
    # plt.imshow(final_foreground_mask)
    # plt.subplot(132)
    # plt.imshow(final_foreground_depth)
    # plt.subplot(133)
    # plt.imshow(humanMasks)
    # plt.show()
    final_foreground_depth = np.zeros_like(foreground_depth)
    final_foreground_mask = np.zeros_like(foreground_mask)
    for hid in np.unique(humanMasks):
        if hid == 0:
            continue
        mask = humanMasks == hid
        if mask.sum() < 200:
            continue
        gt_depth = foreground_depth[mask].flatten()
        temp_depth = depth_prior[mask].flatten()
        if temp_depth.sum() == 0:
            continue    
        mask_gt_nonzero = gt_depth > 0
        
        if np.mean(temp_depth[mask_gt_nonzero]) == 0:
            print('Zero depth')
            continue
        if gt_depth[mask_gt_nonzero].sum() == 0:
            continue
        
        pred_depth_scaled = exposure.match_histograms(temp_depth, gt_depth)

        gt_depth_filled = np.where(gt_depth == 0, pred_depth_scaled, gt_depth)
        # gt_depth_filled = pred_depth_scaled

        
        final_foreground_depth[mask] = gt_depth_filled
        final_foreground_mask[mask] = 1

    image_path = image_paths[camId]
    image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB) / 255.0
    ori_size = image.shape[:2]
    image = cv2.resize(image, (width, height), interpolation=cv2.INTER_LINEAR).reshape(-1, 3)
    
    background_depth_mask_path = background_depth_mask_paths[camId]
    background_depth_mask = np.load(background_depth_mask_path)
    predefined_background_depth_mask = predefined_background_depth_masks[camId]
    background_depth_mask = background_depth_mask & predefined_background_depth_mask
    background_depth = background_depths[camId]
    background_depth = np.where(background_depth_mask, background_depth, 0)
    background_depth = cv2.resize(background_depth, (width, height), interpolation=cv2.INTER_NEAREST)
    background_depth_mask = cv2.resize(background_depth_mask.astype(np.float32), (width, height), interpolation=cv2.INTER_NEAREST).astype(np.bool_)
    background_depth_mask_flat = background_depth_mask.flatten()
    mask = depth_filter(final_foreground_depth, ratio=0.3, max_depth=None)
    final_foreground_mask = final_foreground_mask & mask
    final_foreground_mask_flat = final_foreground_mask.flatten()
    K = dataset.intrinsic_matrices[camId]
    w2c = dataset.w2cs[camId]
    
    c2w = np.linalg.inv(w2c)
    
    foreground_points = depth2points_fun(final_foreground_depth, final_foreground_mask, K)
    foreground_points = c2w[:3, :3] @ foreground_points.T + c2w[:3, 3:4]
    foreground_points = foreground_points.T
    foreground_pcs = np.concatenate([foreground_points, image[final_foreground_mask_flat]], axis=1)
    fuse_foreground_pcs.append(foreground_pcs)
    
    background_points = depth2points_fun(background_depth, background_depth_mask, K)
    background_points = c2w[:3, :3] @ background_points.T + c2w[:3, 3:4]
    background_points = background_points.T
    background_pcs = np.concatenate([background_points, image[background_depth_mask_flat]], axis=1)
    fuse_background_pcs.append(background_pcs)

fuse_foreground_pcs = np.concatenate(fuse_foreground_pcs, axis=0)
fuse_background_pcs = np.concatenate(fuse_background_pcs, axis=0)

pts3d_index = np.arange(fuse_foreground_pcs.shape[0])
pts3d_index_del = []
masks = data['mask']
gt_mask_comb_b = np.where(masks > 0, np.ones_like(masks), np.zeros_like(masks))
for cam_idz in range(num_cam):
    pts3d_index_del.append(bg_filter_fn(w2c=dataset.w2cs[cam_idz],
                K=dataset.intrinsic_matrices[cam_idz],
                pts3d_world=fuse_foreground_pcs[:, :3].T,
                pts3d_index=pts3d_index, 
                masks_comb=gt_mask_comb_b[cam_idz].astype(np.bool_),
                H=height, W=width))
pts3d_index_del = np.concatenate(pts3d_index_del, axis=0)
pts3d_index_del = np.unique(pts3d_index_del)
pts3d_index_keep1 = np.setdiff1d(pts3d_index, pts3d_index_del)
fuse_pcs = np.concatenate([fuse_foreground_pcs[pts3d_index_keep1], fuse_background_pcs], axis=0)

fuse_pcs = np.concatenate([fuse_foreground_pcs, fuse_background_pcs], axis=0)
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(fuse_pcs[:, :3])
pcd.colors = o3d.utility.Vector3dVector(fuse_pcs[:, 3:])
o3d.visualization.draw_geometries([pcd])
o3d.io.write_point_cloud(os.path.join('CVPR2025/point_cloud', 'point_cloud_ours.ply'), pcd)