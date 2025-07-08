import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import cv2
import glob
import torch
import tqdm
import time
import copy 
import numpy as np
import open3d as o3d
from PIL import Image
import torch.nn.functional as F
import matplotlib.pyplot as plt
from transformers import pipeline
from dataset.wildtrack import Wildtrack, SuperPixels
from torch.utils.data import DataLoader
from ProteusLib import SuperPixelGaussian, render, depth_constraints, DisparityToDepth, pearson_correlation_loss, init_splats_from_ply, rasterize_splats, init_splats_from_pth
from argparse import ArgumentParser
from utils.argument_utils import OptimizationParams
from dataset.wildtrack import Wildtrack, SuperPixels, WildtrackDetection, frameDataset
from unsupervised_cluster import bevMap2StandingPoint
from submodules.gsplat.strategy import DefaultStrategy
from s04_gs_optimize import bg_filter_fn
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor

def to_numpy(x):
    '''
        Only for pytorch tensor or numpy array
    '''
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return x

def empty_cache():
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    
def mask_filter(masks, mask_min_area):
    for id, mask in enumerate(masks):
        for mask_id in torch.unique(mask):
            if mask_id == 0:
                continue
            mask_area = torch.sum(mask == mask_id)
            if mask_area < mask_min_area:
                masks[id][mask == mask_id] = 0
    return masks


def step01_GS_init_depth_prior(frame_idx, data_root, detect_data, data, round_name='1_1'):
    print(f'Processing frame {frame_idx}')
    num_cam = 7
    ground_range =  [0, 0, 0, 480, 1440, 80]
    ground_range[0] = -300 + 2.5 * ground_range[0]
    ground_range[3] = -300 + 2.5 * ground_range[3]
    
    ground_range[1] = -900 + 2.5 * ground_range[1]
    ground_range[4] = -900 + 2.5 * ground_range[4]
    init_gs_path = os.path.join('/home/jiahao/Downloads/data/wildtrack_data', 'depths', frame_idx)
    gs_pcd = o3d.io.read_point_cloud(os.path.join(init_gs_path, f'gs_2_1.ply'))
    gs_pcd_xyz = np.array(gs_pcd.points)
    within_mask = (gs_pcd_xyz[:, 0] > ground_range[0]) & (gs_pcd_xyz[:, 0] < ground_range[3]) & (gs_pcd_xyz[:, 1] > ground_range[1]) & (gs_pcd_xyz[:, 1] < ground_range[4])
    gs_pcd_xyz = gs_pcd_xyz[~within_mask]
    
    x_intvl = 15
    y_intvl = 15
    num_sample = 20
    map_gt = detect_data[1]
    coord_xyz = bevMap2StandingPoint(map_gt.squeeze(0).cpu().numpy())


    all_pts3d_xyz = []

    for coord in coord_xyz:
        x_min = coord[0] - x_intvl
        x_max = coord[0] + x_intvl
        y_min = coord[1] - y_intvl
        y_max = coord[1] + y_intvl
        z_min = 0
        z_max = 180
        x = np.linspace(x_min, x_max, num_sample)
        y = np.linspace(y_min, y_max, num_sample)
        z = np.linspace(z_min, z_max, num_sample*5)
        pts3d_xyz = np.stack(np.meshgrid(x, y, z, indexing='ij'), axis=-1).reshape(-1, 3)
        all_pts3d_xyz.append(pts3d_xyz) 
    all_pts3d_xyz = np.concatenate(all_pts3d_xyz, axis=0)
    all_pts3d_xyz = np.concatenate([all_pts3d_xyz, gs_pcd_xyz], axis=0)
    
    H, W = data['H'].item(), data['W'].item()
    c2ws = torch.cat([cam['c2w']for cam in  data['novel_view']], dim=0).cuda()
    w2cs = torch.cat([cam['w2c']for cam in  data['novel_view']], dim=0).cuda()
    Ks = torch.cat([cam['K']for cam in  data['novel_view']], dim=0).cuda()
    masks = data['mask'].squeeze(0).cpu().numpy().astype(np.bool_) 
    masks = np.where(masks > 0, np.ones_like(masks), np.zeros_like(masks)) # (7, h, w)
    
    pts3d_index = np.arange(all_pts3d_xyz.shape[0])
    pts3d_index_del = []
    for cam_idz in range(num_cam):
        pts3d_index_del.append(bg_filter_fn(w2c=w2cs[cam_idz].cpu().numpy(),
                    K=Ks[cam_idz].cpu().numpy(),
                    pts3d_world=all_pts3d_xyz.T,
                    pts3d_index=pts3d_index, 
                    masks_comb=masks[cam_idz],
                    H=H, W=W))
       
    pts3d_index_del = np.concatenate(pts3d_index_del, axis=0)
    pts3d_index_del = np.unique(pts3d_index_del)
    pts3d_index_keep1 = np.setdiff1d(pts3d_index, pts3d_index_del)
    all_pts3d_xyz = all_pts3d_xyz[pts3d_index_keep1]
    all_pts3d_rgb = np.zeros_like(all_pts3d_xyz)
    
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(all_pts3d_xyz)
    pcd.colors = o3d.utility.Vector3dVector(all_pts3d_rgb)
    
    save_root = os.path.join(data_root, 'depths')
    if not os.path.exists(save_root):
        os.makedirs(save_root)
    save_folder = os.path.join(save_root, frame_idx)
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    o3d.io.write_point_cloud(os.path.join(save_folder, f'init_gs_{round_name}.ply'), pcd)
    
    return all_pts3d_xyz, all_pts3d_rgb, save_folder

def depth_init(
         frame_idx,
         data_root, 
         data,
         round_name='1_1',
         dataset=None 
         ):
    round_idx = int(round_name.split('_')[0])
    # gt_mask = data['mask'].permute(1, 0, 2, 3)
    masks = data['mask'].squeeze(0).cuda()
    gt_mask_comb_b = torch.where(masks > 0, torch.ones_like(masks), torch.zeros_like(masks)) # (7, h, w)
    gt_mask_split = masks
    # gt_mask_comb_f = gt_mask_comb_b.float()
    gt_mask_comb_b = gt_mask_comb_b.bool()
    masks_split = data['mask'].squeeze(0)
    # masks_comb = torch.where(masks_split>0, torch.ones_like(masks_split), torch.zeros_like(masks_split)).cpu().numpy()
    # masks_comb = masks_comb.astype(np.bool_)
    c2ws = torch.cat([cam['c2w']for cam in  data['novel_view']], dim=0).cuda()
    w2cs = torch.cat([cam['w2c']for cam in  data['novel_view']], dim=0).cuda()
    Ks = torch.cat([cam['K']for cam in  data['novel_view']], dim=0).cuda()
    H, W = data['H'].item(), data['W'].item()
    
    # init model
    save_folder = os.path.join(data_root, 'depths', frame_idx)
    super_pixel_ply = os.path.join(save_folder, f'init_gs_{round_name}.ply')
    assert os.path.exists(super_pixel_ply), f'Fail to find {super_pixel_ply}'

    splats, _ = init_splats_from_ply(super_pixel_ply, dataset.scene_scale, with_cluster_feat=False)
    init_splats = copy.deepcopy(splats)
    init_splats['opacities'] = torch.ones_like(splats['opacities']) * 100
    init_splats['scales'] = torch.ones_like(splats['scales']) * splats['scales'].median() * 0.1
    renders, _, _ = rasterize_splats(init_splats , w2cs, Ks, W, H, render_mode="RGB+D", sh_degree=3, rasterize_feat=False)
    init_depths = renders[..., 3:4]

    sup_depths = []
    for cam_idx in range(init_depths.shape[0]):
        init_depth = init_depths[cam_idx]
        
        cam_mask = masks_split[cam_idx]
        sup_depth = torch.zeros_like(init_depth)
        for h_id in np.unique(cam_mask):
            if h_id == 0:
                continue
            mask = cam_mask == h_id
            row, col = torch.where(mask)
            for r in torch.unique(row):
                
                row_depth = init_depth[row[row==r], col[row==r]]
                row_depth = row_depth[row_depth > 0]
                if len(row_depth) == 0 or len(row_depth) < 5:
                    continue
                median_depth = torch.median(row_depth)
                sup_depth[row[row==r], col[row==r]] = median_depth.float()
            # find the region with high depth variance
            mask_depth = sup_depth[mask]
            if mask_depth.numel() < 5:
                sup_depth[mask] = 0
            else:
                mean_depth = torch.mean(mask_depth) # not mean, should be median
                std_depth = torch.std(mask_depth)
                t_low = mean_depth - 1.5 * std_depth
                t_high = mean_depth +  1.5 * std_depth
                sup_depth[mask] = torch.where((mask_depth > t_low) & (mask_depth < t_high), mask_depth, 0)
            
        sup_depths.append(sup_depth)
    sup_depths = torch.stack(sup_depths, dim=0)
    return sup_depths.detach().cpu(), c2ws.detach().cpu(), w2cs.detach().cpu(), Ks.detach().cpu(), H, W, gt_mask_comb_b.detach().cpu(), gt_mask_split.detach().cpu()
    

def depth_filter(depths, c2ws, w2cs, Ks, H, W, masks):
    all_pts3d = []
    all_pts3d_uv = []
    num_cam = depths.shape[0]
    for cam_idx in range(num_cam):
        depth = depths[cam_idx].squeeze()
        c2w = c2ws[cam_idx]
        K = Ks[cam_idx]
        # v = torch.arange(0, H).view(-1, 1).expand(-1, W).float()
        # u = torch.arange(0, W).view(1, -1).expand(H, -1).float()
        v = np.arange(0, H).reshape(-1, 1).repeat(W, axis=1).astype(np.float32)
        u = np.arange(0, W).reshape(1, -1).repeat(H, axis=0).astype(np.float32)
        x = (u - K[0, 2]) / K[0, 0] * depth
        y = (v - K[1, 2]) / K[1, 1] * depth
        z = depth.reshape(-1)   
        pts3d = np.stack([x.flatten(), y.flatten(), z], axis=-1).reshape(-1, 3)
        pts3d = c2w[:3, :3] @ pts3d.T + c2w[:3, 3:4]
        pts3d = pts3d.T[z>0]
        uvs = np.stack([np.ones_like(u.flatten()) * cam_idx, u.flatten(), v.flatten()], axis=-1).reshape(-1, 3)
        uvs = uvs[z>0]
        all_pts3d.append(pts3d)
        all_pts3d_uv.append(uvs)
    
    all_pts3d_uv  = np.concatenate(all_pts3d_uv, axis=0)
    all_pts3d_xyz = np.concatenate(all_pts3d, axis=0)
    pts3d_index = np.arange(all_pts3d_xyz.shape[0])
    pts3d_index_del = []
    for cam_idz in range(num_cam):
        pts3d_index_del.append(bg_filter_fn(w2c=w2cs[cam_idz],
                    K=Ks[cam_idz],
                    pts3d_world=all_pts3d_xyz.T,
                    pts3d_index=pts3d_index, 
                    masks_comb=masks[cam_idz],
                    H=H, W=W))
       
    pts3d_index_del = np.concatenate(pts3d_index_del, axis=0)
    pts3d_index_del = np.unique(pts3d_index_del)
    pts3d_index_keep1 = np.setdiff1d(pts3d_index, pts3d_index_del)
    all_pts3d_xyz = all_pts3d_xyz[pts3d_index_keep1]
    all_pts3d_uv = all_pts3d_uv[pts3d_index_keep1]
    
    # filter depth
    filter_depths = []
    for cam_idx in range(num_cam):
        uv = all_pts3d_uv[all_pts3d_uv[:, 0] == cam_idx][:, 1:].astype(np.int32)
        depth = depths[cam_idx].squeeze()
        filter_depth_mask = np.zeros_like(depth).astype(np.bool_)
        filter_depth_mask[uv[:, 1], uv[:, 0]] = True
        filter_depth = np.zeros_like(depth)
        filter_depth[filter_depth_mask] = depth[filter_depth_mask]
        filter_depths.append(filter_depth)
    filter_depths = np.stack(filter_depths, axis=0)
    
    return filter_depths, all_pts3d_xyz

def depth_sup(depths, c2ws, w2cs, Ks, H, W, masks_comb, masks_split, round_name, save_folder):
    # first round filter depth
    filter_depths, all_pts3d_xyz = depth_filter(to_numpy(depths), 
                                                to_numpy(c2ws), 
                                                to_numpy(w2cs), 
                                                to_numpy(Ks), 
                                                to_numpy(H), 
                                                to_numpy(W), 
                                                to_numpy(masks_comb))
    # supplement depth
    sup_depths = []
    num_cam = c2ws.shape[0]
    for cam_idx in range(depths.shape[0]):
        sup_depth = np.zeros_like(filter_depths[cam_idx])
        mask_split = masks_split[cam_idx]
        for h_id in np.unique(mask_split):
            if h_id == 0:
                continue
            mask = mask_split == h_id
            if mask.sum() < 100:
                continue
            temp_depth = copy.deepcopy(filter_depths[cam_idx])
            temp_depth[~mask] = 0
            rows, cols = np.where(mask)
            for row in np.unique(rows):
                row_depth = temp_depth[rows[rows==row], cols[rows==row]]
                row_depth = row_depth[row_depth > 0]
                if row_depth.size == 0:
                    continue
                median_depth = np.median(row_depth)
                sup_depth[rows[rows==row], cols[rows==row]] = median_depth
        sup_depths.append(sup_depth)
    sup_depths = np.stack(sup_depths, axis=0)
    
    all_pts3d = []
    all_pts3d_uv = []
    for cam_idx in range(num_cam):
        depth = torch.Tensor(sup_depths[cam_idx]).squeeze()
        c2w = c2ws[cam_idx]
        K = Ks[cam_idx]
        v = torch.arange(0, H).view(-1, 1).expand(-1, W).float()
        u = torch.arange(0, W).view(1, -1).expand(H, -1).float()
        x = (u - K[0, 2]) / K[0, 0] * depth
        y = (v - K[1, 2]) / K[1, 1] * depth
        z = depth.reshape(-1)   
        pts3d = torch.stack([x.flatten(), y.flatten(), z], dim=-1).reshape(-1, 3)
        pts3d = c2w[:3, :3] @ pts3d.T + c2w[:3, 3:4]
        pts3d = pts3d.T[z>0]
        uvs = torch.stack([torch.ones_like(u.flatten()) * cam_idx, u.flatten(), v.flatten()], dim=-1).reshape(-1, 3)
        uvs = uvs[z>0]
        all_pts3d.append(pts3d)
        all_pts3d_uv.append(uvs)
    
    all_pts3d_uv  = torch.cat(all_pts3d_uv, dim=0).cpu().numpy()
    all_pts3d_xyz = torch.cat(all_pts3d, dim=0).cpu().numpy()
    
    save_depth_path = os.path.join(save_folder, f'depths_{round_name}.npy')
    np.save(save_depth_path, sup_depths)
    sup_depths_png = np.concatenate(sup_depths, axis=1).squeeze()
    save_depth_path = os.path.join(save_folder, f'depths_{round_name}.png')
    # plt.imshow(sup_depths_png)
    # plt.axis('off')
    # plt.savefig(save_depth_path, bbox_inches='tight', pad_inches=0, dpi=300)
    # plt.close()
    # save by cv2
    cv2.imwrite(save_depth_path, sup_depths_png)
    
    
    save_pcd_path = os.path.join(save_folder,  f'init_gs_{round_name}.ply')
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(all_pts3d_xyz)
    o3d.io.write_point_cloud(save_pcd_path, pcd)

    filter_depths, filter_all_pts3d_xyz = depth_filter(to_numpy(sup_depths), 
                                                       to_numpy(c2ws), 
                                                       to_numpy(w2cs), 
                                                       to_numpy(Ks), 
                                                       to_numpy(H), 
                                                       to_numpy(W), 
                                                       to_numpy(masks_comb))
    save_depth_path = os.path.join(save_folder, f'filter_depths_{round_name}.npy')
    np.save(save_depth_path, filter_depths)
    filter_depths_png = np.concatenate(filter_depths, axis=1).squeeze()
    save_depth_path = os.path.join(save_folder, f'filter_depths_{round_name}.png')
    # plt.imshow(filter_depths_png)
    # plt.axis('off')
    # plt.savefig(save_depth_path, bbox_inches='tight', pad_inches=0, dpi=300)
    # plt.close()
    cv2.imwrite(save_depth_path, filter_depths_png)
    save_pcd_path = os.path.join(save_folder,  f'filter_gs_{round_name}.ply')
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(filter_all_pts3d_xyz)
    # o3d.visualization.draw_geometries([pcd])
    o3d.io.write_point_cloud(save_pcd_path, pcd)
    
def process_frame(data, wildtrack_detect_data, wildtrack_data, data_root, args):
    frame_idx = data['frame_idx'][0]
    detect_data_idx = int(frame_idx) // 5

    _, _, save_folder = step01_GS_init_depth_prior(frame_idx=frame_idx,
                                                   data_root=data_root, 
                                                   detect_data=wildtrack_detect_data[detect_data_idx], 
                                                   data=data,
                                                   round_name=args.round)
    
    sup_depths, c2ws, w2cs, Ks, H, W, masks_comb, mask_split = depth_init(frame_idx=frame_idx, 
                                                                         data_root=data_root, 
                                                                         data=data, 
                                                                         round_name=args.round,
                                                                         dataset=wildtrack_data)
    
    depth_sup(sup_depths, c2ws, w2cs, Ks, H, W, masks_comb, mask_split, args.round, save_folder)
    print(f'Finish processing frame {frame_idx}')
    
def main():
    # first round
    args_src = [
                '--root', '/home/jiahao/Downloads/data/wildtrack_data_gt',
                '--round', '1_1',
                '--n-segments', '30',
                '--start-with', '0',
                '--end-with', '-1',#'-1',
                '--init_opacity', '0.1',
                "--bg_filter", "--no-depth_filter",
            ]

    parser = ArgumentParser("Per scene training using Gaussian Splatting", add_help=True)
    parser.add_argument("--root", type=str, required=True, help="path to image folder")
    parser.add_argument("--round", type=str, required=True, help="round name")
    parser.add_argument("--start-with", type=int, default=0, help="the index of the first image to start with")
    parser.add_argument("--end-with", type=int, default=-1, help="the index of the last image to end with")
    parser.add_argument("--n-segments", type=int, default=30, help="the number of superpixels for each person")
    parser.add_argument("--init_opacity", type=float, default=0.1, help="the initial opacity of the gaussian")
    
    parser.add_argument("--bg_filter", action='store_true', help="whether to filter out the points out of the mask")
    parser.add_argument("--no-bg_filter", action='store_false', dest='bg_filter', help="Disable the bg_filter")

    parser.add_argument("--depth_filter", action='store_true', help="whether to filter out the points out of the mask based on depth")
    parser.add_argument("--no-depth_filter", action='store_false', dest='depth_filter', help="Disable the depth_filter")
    
    args = parser.parse_args(args_src) # FOR DEBUG
    # args = parser.parse_args()
    print(args)
    
    # Init dataset and dataloader
    data_root = args.root
    n_segments = args.n_segments
    start_with = args.start_with
    end_with = args.end_with

    print(f'The start_with and end_with are: {start_with} and {end_with}')

    wildtrack_detect_data = frameDataset(WildtrackDetection('/home/jiahao/Downloads/data/Wildtrack'))
    wildtrack_data = Wildtrack(data_root, mask_type='people', mask_label_type='split', start_with=start_with, end_with=end_with)
    wildtrack_dataloader = DataLoader(wildtrack_data, batch_size=1, pin_memory=True, num_workers=4, drop_last=True)
    sp_dataset = SuperPixels(data_root, n_segments=n_segments, HW=wildtrack_data.image_feat_size, start_with=start_with, end_with=end_with)
    assert len(wildtrack_data) == len(sp_dataset), f'Length of wildtrack_data and sp_dataset are not equal: {len(wildtrack_data)} and {len(sp_dataset)}'

    print('Start to generate matching label...')

    # Use ThreadPoolExecutor for multithreading
    with ThreadPoolExecutor(max_workers=4) as executor:  # Adjust the max_workers based on your CPU cores
        futures = []
        for idx, data in enumerate(wildtrack_dataloader):
            future = executor.submit(process_frame, data, wildtrack_detect_data, wildtrack_data, data_root, args)
            futures.append(future)
        
        # Wait for all threads to complete
        concurrent.futures.wait(futures)

    print('All frames processed.')
        
if __name__ == '__main__':
    main()