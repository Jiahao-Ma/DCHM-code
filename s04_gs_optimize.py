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

def depth_rank_loss(pred_depth, gt_depth, split_masks, sample_n=128, ranomd_mask_num=10):
    final_loss = 0
    mask_num = 0
    for cam_id in range(len(split_masks)):
        split_mask = split_masks[cam_id]
        hum_ids = np.random.choice(torch.unique(split_mask).detach().cpu().numpy(), ranomd_mask_num)
        hum_ids = hum_ids[hum_ids > 0]
        for hum_id in hum_ids:
            pd = pred_depth[cam_id][split_mask == hum_id]
            gd = gt_depth[cam_id][split_mask == hum_id]
            half_sample_n = np.minimum(sample_n, pd.shape[0]) // 2
            selected_idxs = np.random.choice(pd.shape[0], half_sample_n*2, replace=False)        
            sample_idx0 = selected_idxs[:half_sample_n]
            sample_idx1 = selected_idxs[half_sample_n:]
            pd0, pd1 = pd[sample_idx0], pd[sample_idx1]
            gd0, gd1 = gd[sample_idx0], gd[sample_idx1]
            # note that we use dpt mono depth (not disparity)
            mask = torch.where(gd0 > gd1, True, False)
            d0 = pd0 - pd1
            d1 = pd1 - pd0

            depth_loss0 = torch.zeros_like(d0)
            depth_loss0[mask] += d1[mask]
            depth_loss0[~mask] += d0[~mask]
            final_loss += torch.mean(torch.clamp(depth_loss0, min=0.0))
            mask_num += 1
            
    return final_loss / mask_num

def step01_GS_init_us(frame_idx, data_root, data, num_depth_bin, superpixel_data, round_name='1_1', mask_min_area=-1):
    H, W = data['H'].cuda(), data['W'].cuda()
    c2ws = torch.cat([cam['c2w']for cam in  data['novel_view']], dim=0).cuda()
    w2cs = torch.cat([cam['w2c']for cam in  data['novel_view']], dim=0).cuda()
    Ks = torch.cat([cam['K']for cam in  data['novel_view']], dim=0).cuda()
    masks = data['mask'].squeeze(0).cuda()
    if mask_min_area > 0:
        masks = mask_filter(masks, mask_min_area) 
    masks = torch.where(masks > 0, torch.ones_like(masks), torch.zeros_like(masks)) # (7, h, w)
    
    all_images = torch.from_numpy(np.stack(superpixel_data['image'], axis=0)).cuda()
        
    save_root = os.path.join(data_root, 'depths')
    if not os.path.exists(save_root):
        os.makedirs(save_root)
    save_folder = os.path.join(save_root, frame_idx)
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    if not os.path.exists(os.path.join(save_folder, f'init_gs_{round_name}.ply')):
        num_cam = 7
        unit_scale = 100 # cm to m
        max_depth = 40.0
        rel_depth = torch.linspace(0.05, 1, num_depth_bin, device=w2cs.device)
        rel_depth = rel_depth * unit_scale * max_depth
        
        all_pts3d_xyz = []
        all_pts3d_rgb = []
        
        for cam_idx in range(num_cam):
            K = Ks[cam_idx]
            c2w = c2ws[cam_idx]
            # load centroid 
            centroids = torch.from_numpy(superpixel_data['centroids'][cam_idx]).cuda().to(dtype=torch.float32)
            if centroids.shape[0] == 0:
                continue
            # load image
            img = all_images[cam_idx]
            
            # extract the rgb color by bilinear interpolation
            cnt = centroids.to(dtype=torch.int32)
            pts3d_rgb = img[cnt[:, 0], cnt[:, 1]]
            pts3d_rgb = pts3d_rgb[:, None, :].expand(-1, rel_depth.shape[0], -1).contiguous().view(-1, 3)
            
            x = (centroids[:, 1] - K[0, 2]) / K[0, 0] 
            y = (centroids[:, 0] - K[1, 2]) / K[1, 1]
            x = x.view(-1, 1) * rel_depth.view(1, -1)
            y = y.view(-1, 1) * rel_depth.view(1, -1)
            z = rel_depth.view(1, -1).expand(x.shape[0], -1)
            pts3d = torch.stack([x, y, z], dim=-1).contiguous().view(-1, 3) # (N, 3)
            pts3d_world = c2w[:3, :3] @ pts3d.T + c2w[:3, 3:4]
            
            # Filter out the points out of the mask
            for cam_idy in range(num_cam):
                if cam_idx == cam_idy:
                    continue
                pts3d_index = torch.arange(pts3d_world.shape[1], device=pts3d_world.device)
                human_mask = masks[cam_idy].to(dtype=torch.bool)
                w2c = w2cs[cam_idy]
                K = Ks[cam_idy]
                pts3d_cam = w2c[:3, :3] @ pts3d_world + w2c[:3, 3:4]
                pts3d_scn = K @ pts3d_cam
                pts3d_scn = pts3d_scn[:2, :] / pts3d_scn[2, :]
                pts3d_scn = pts3d_scn.to(dtype=torch.int32)
                # if pts3d_scn's pixel in the mask and within the image, then it is valid
                mask1 = (pts3d_scn[0, :] >= 0) & (pts3d_scn[0, :] < W) & (pts3d_scn[1, :] >= 0) & (pts3d_scn[1, :] < H)
                pts3d_scn = pts3d_scn[:, mask1]
                mask2 = human_mask[pts3d_scn[1, :], pts3d_scn[0, :]]
                pts3d_scn = pts3d_scn[:, mask2]
                
                valid_index = torch.cat([pts3d_index[mask1][mask2], pts3d_index[~mask1]])
                pts3d_world = pts3d_world[:, valid_index]
                pts3d_rgb = pts3d_rgb[valid_index]
            
            all_pts3d_xyz.append(pts3d_world.T)
            all_pts3d_rgb.append(pts3d_rgb)
            
        all_pts3d_xyz = torch.cat(all_pts3d_xyz, dim=0)
        all_pts3d_rgb = torch.cat(all_pts3d_rgb, dim=0)   
        
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(all_pts3d_xyz.cpu().numpy())
        pcd.colors = o3d.utility.Vector3dVector(all_pts3d_rgb.cpu().numpy())
        o3d.io.write_point_cloud(os.path.join(save_folder, f'init_gs_{round_name}.ply'), pcd)
    else:
        pcd = o3d.io.read_point_cloud(os.path.join(save_folder, f'init_gs_{round_name}.ply'))
        all_pts3d_xyz = torch.from_numpy(np.asarray(pcd.points)).cuda()
        all_pts3d_rgb = torch.from_numpy(np.asarray(pcd.colors)).cuda()
    return all_pts3d_xyz, all_pts3d_rgb, all_images

def step01_GS_init_depth_prior(frame_idx, data_root, detect_data, data, superpixel_data, depth_threshold=3, round_name='1_1', rebuild=True, bg_filter=True):
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
    
    x_intvl = 30
    y_intvl = 30
    num_sample = 20
    map_gt = detect_data[1]
    coord_xyz = bevMap2StandingPoint(map_gt.squeeze(0).cpu().numpy())
    
    all_pts3d_xyz = []
    meshes = []
    for coord in coord_xyz:
        x_min = coord[0] - x_intvl
        x_max = coord[0] + x_intvl
        y_min = coord[1] - y_intvl
        y_max = coord[1] + y_intvl
        z_min = 0
        z_max = 160
        x = np.linspace(x_min, x_max, num_sample)
        y = np.linspace(y_min, y_max, num_sample)
        z = np.linspace(z_min, z_max, num_sample*5)
        pts3d_xyz = np.stack(np.meshgrid(x, y, z, indexing='ij'), axis=-1).reshape(-1, 3)
        all_pts3d_xyz.append(pts3d_xyz) 
    all_pts3d_xyz = np.concatenate(all_pts3d_xyz, axis=0)
    all_pts3d_xyz = np.concatenate([all_pts3d_xyz, gs_pcd_xyz], axis=0)
    
    
    all_images = torch.from_numpy(np.stack(superpixel_data['image'], axis=0)).cuda()
    
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
    
    return all_pts3d_xyz, all_pts3d_rgb, all_images

def step02_train(
         train_args,
         frame_idx,
         data_root, 
         data,
         all_images,
         round_name='1_1',
         remove_init_gs=True,
         dataset=None,
         save_folder = None,
         super_pixel_ply = None
         ):
    round_idx = int(round_name.split('_')[0])
    # gt_mask = data['mask'].permute(1, 0, 2, 3)
    masks = data['mask'].squeeze(0).cuda()
    gt_mask_comb_b = torch.where(masks > 0, torch.ones_like(masks), torch.zeros_like(masks)) # (7, h, w)
    gt_mask_split = masks
    gt_mask_comb_f = gt_mask_comb_b.float()
    gt_mask_comb_b = gt_mask_comb_b.bool().unsqueeze(-1)
    
    # masks_comb = torch.where(masks_split>0, torch.ones_like(masks_split), torch.zeros_like(masks_split)).cpu().numpy()
    # masks_comb = masks_comb.astype(np.bool_)
    c2ws = torch.cat([cam['c2w']for cam in  data['novel_view']], dim=0).cuda()
    w2cs = torch.cat([cam['w2c']for cam in  data['novel_view']], dim=0).cuda()
    Ks = torch.cat([cam['K']for cam in  data['novel_view']], dim=0).cuda()
    H, W = data['H'].item(), data['W'].item()
    
    # init super pixel data
    gt_imgs = all_images.to(dtype=torch.float32)
    
    depth_anything_depths = np.load(os.path.join('submodules/Depth-Anything-V2/data/wildtrack', frame_idx, 'pred_depths.npy'))
    depthpro_depths = np.load(os.path.join('submodules/depthpro/data/wildtrack', frame_idx, 'pred_depths.npy'))
    mono_depths = []
    for i in range(depth_anything_depths.shape[0]):
        if i == 2:
            mono_depths.append(cv2.resize(depth_anything_depths[i], (960, 540), interpolation=cv2.INTER_NEAREST))
        else:
            mono_depths.append(cv2.resize(depthpro_depths[i], (960, 540), interpolation=cv2.INTER_NEAREST))
            
    mono_depths = np.stack(mono_depths, axis=0)
    mono_depths = torch.from_numpy(mono_depths).cuda().unsqueeze(-1)
    # init arguments
    parser = ArgumentParser(description="Training script parameters")
    op = OptimizationParams(parser)
    
    args = parser.parse_args([])
    opt = op.extract(args)
    if save_folder is None:
    # init model
        save_folder = os.path.join(data_root, 'depths', frame_idx)
        super_pixel_ply = os.path.join(save_folder, f'init_gs_{round_name}.ply')
    assert os.path.exists(super_pixel_ply), f'Fail to find {super_pixel_ply}'
    # model = SuperPixelGaussian(super_pixel_ply, max_sh_degree=3, init_opacity=train_args.init_opacity)
    # model.training_setup(opt)
    
    splats, optimizers = init_splats_from_ply(super_pixel_ply, dataset.scene_scale, with_cluster_feat=False)
    gt_depths = np.load(os.path.join(os.path.dirname(save_folder), 'depths_1_1.npy'))
    gt_depths = torch.from_numpy(gt_depths).cuda().unsqueeze(-1)
    strategy = DefaultStrategy(refine_start_iter=100, refine_stop_iter=train_args.epochs, reset_every=2000, refine_every=300, verbose=True)
    strategy.check_sanity(splats, optimizers)
    strategy_state = strategy.initialize_state(scene_scale=dataset.scene_scale)
    schedulers = [
        # means has a learning rate schedule, that end at 0.01 of the initial value
        torch.optim.lr_scheduler.ExponentialLR(
            optimizers["means"], gamma=0.01 ** (1.0 / train_args.epochs)
        ),
    ]
    # hyperparameters
    loss_bg_w = train_args.loss_bg_w       
    loss_rgb_w = train_args.loss_rgb_w
    
    loss_global_depth_w = train_args.loss_global_depth_w
    loss_local_depth_w = train_args.loss_local_depth_w
    loss_quantization_w = train_args.loss_quantization_w
    loss_rank_depth_w = train_args.loss_rank_depth_w

    pbar = tqdm.tqdm(range(train_args.epochs))
    for epoch in pbar:
        losses = []
        
        renders, pred_alphas, info = rasterize_splats(splats , w2cs, Ks, W, H, render_mode="RGB+D", sh_degree=3, rasterize_feat=False)
        pred_imgs, pred_metric_depth = renders[..., 0:3], renders[..., 3:4]
        strategy.step_pre_backward(
                params=splats,
                optimizers=optimizers,
                state=strategy_state,
                step=epoch,
                info=info,
            )
        
        # global depth constraints
        if loss_global_depth_w > 0 and gt_depths is not None:
            loss_global_depth = 0
            for cam_id, cam_mask in enumerate(gt_mask_comb_b):
                pred_depth = pred_metric_depth[cam_id][cam_mask] 
                gt_depth = gt_depths[cam_id][cam_mask]
                # randomly select 10% of the pixels
                mask = gt_depth > 1e1
                loss_global_depth = loss_global_depth + torch.nn.functional.mse_loss(pred_depth[mask], gt_depth[mask])
            loss_global_depth = loss_global_depth / len(gt_mask_comb_b)
           
        else:
            loss_global_depth = torch.tensor(0)
        
        # local depth constraints
        if loss_local_depth_w > 0:
            loss_local_depth = depth_constraints(pred_metric_depth, gt_mask_split, data['novel_view'], loss_type=train_args.depth_constraint, random_select=train_args.random_select)
        else:
            loss_local_depth = torch.tensor(0)
            
        # # depth rank loss
        if loss_rank_depth_w > 0:
            loss_rank_depth = depth_rank_loss(pred_metric_depth, mono_depths, gt_mask_split)
        else:
            loss_rank_depth = torch.tensor(0)
            
        # mse loss
        loss_rgb = torch.nn.functional.mse_loss(pred_imgs, gt_imgs)
        
        # bg loss
        loss_bg = torch.nn.functional.l1_loss(pred_alphas.squeeze(-1), gt_mask_comb_f)
        
        
        loss_sparsity = torch.tensor(0)
        loss_quantization = torch.tensor(0)
        loss = loss_rgb * loss_rgb_w + loss_bg * loss_bg_w + loss_local_depth * loss_local_depth_w + loss_global_depth * loss_global_depth_w + loss_rank_depth * loss_rank_depth_w
        
        losses.append(loss.detach().cpu().numpy())
        loss.backward(retain_graph=True)
        
        strategy.step_post_backward(
                params=splats,
                optimizers=optimizers,
                state=strategy_state,
                step=epoch,
                info=info,
            )
        for optimizer in optimizers.values():
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        for scheduler in schedulers:
            scheduler.step()
        
        rgb_loss_print = loss_rgb.detach().cpu().numpy() * loss_rgb_w
        bg_loss_print = loss_bg.detach().cpu().numpy() * loss_bg_w
        local_depth_loss_print = loss_local_depth.detach().cpu().numpy() * loss_local_depth_w
        global_depth_loss_print = loss_global_depth.detach().cpu().numpy() * loss_global_depth_w
        quantization_loss_print = loss_quantization.detach().cpu().numpy() * loss_quantization_w
        rank_depth_loss_print = loss_rank_depth.detach().cpu().numpy() * loss_rank_depth_w
        pbar.set_description(f'Epoch {epoch} Loss local dc: {local_depth_loss_print:6.6f} | Loss global dc: {global_depth_loss_print:6.6f} | Loss bg: {bg_loss_print:6.6f} | Loss rgb: {rgb_loss_print:6.6f} | Loss depth rank: {rank_depth_loss_print:6.6f} | Loss: {np.mean(losses):.6f}')
        pbar.update()
        
        
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        
    data = {"splats": splats.state_dict()}
    torch.save(data, os.path.join(save_folder, f'gs_{round_name}.pth'))
    
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(splats['means'].detach().cpu().numpy())
    o3d.io.write_point_cloud(os.path.join(save_folder, f'gs_{round_name}.ply'), pcd)
    return splats, os.path.join(save_folder, f'gs_{round_name}.pth')

def bg_filter_fn(w2c, K, pts3d_world, pts3d_index, masks_comb, H, W):
    pts3d_cam = w2c[:3, :3] @ pts3d_world + w2c[:3, 3:4]
    pts3d_scrn = K @ pts3d_cam
    pts3d_scrn = pts3d_scrn[:2, :] / (pts3d_scrn[2:3, :] + 1e-10)
    mask1 = (pts3d_scrn[0, :] > 0) & (pts3d_scrn[0, :] < W) & (pts3d_scrn[1, :] > 0) & (pts3d_scrn[1, :] < H)
    pts3d_scrn = pts3d_scrn[:, mask1]
    mask2 = masks_comb[pts3d_scrn[1].astype(np.int32), pts3d_scrn[0].astype(np.int32)]
    return pts3d_index[mask1][~mask2]

def depth_filter_fn(w2c, K, pts3d_world, pts3d_index, H, W, nearby_depth, depth_offset_threshold):
    pts3d_cam = w2c[:3, :3] @ pts3d_world + w2c[:3, 3:4]
    render_depth = pts3d_cam[-1, :]
    pts3d_scrn = K @ pts3d_cam
    pts3d_scrn = pts3d_scrn[:2, :] / pts3d_scrn[2:3, :]
    mask1 = (pts3d_scrn[0, :] > 0) & (pts3d_scrn[0, :] < W) & (pts3d_scrn[1, :] > 0) & (pts3d_scrn[1, :] < H)
    pts3d_scrn = pts3d_scrn[:, mask1]
    nearby_depth = nearby_depth[pts3d_scrn[1].astype(np.int32), pts3d_scrn[0].astype(np.int32)]
    depth_dist = np.abs(nearby_depth - render_depth[mask1])
    mask2 = depth_dist < depth_offset_threshold
    return pts3d_index[mask1][mask2]

def depth_filter_grad(depth, ratio=0.5, max_depth=30):
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

    return mask

def step03_generate_matching_label(ckpt_path, data, superpixel_data, num_cam = 7, depth_offset_threshold=100, bg_filter=True, depth_filter=True, round_name='1_1'):
    
    splats = init_splats_from_pth(ckpt_path)
    # splats['opacities'] = torch.ones_like(splats['opacities']) * 100
    masks_split = data['mask'].squeeze(0)
    masks_comb = torch.where(masks_split>0, torch.ones_like(masks_split), torch.zeros_like(masks_split)).cpu().numpy()
    masks_comb = masks_comb.astype(np.bool_)
    c2ws = torch.cat([cam['c2w']for cam in  data['novel_view']], dim=0).cuda()
    w2cs = torch.cat([cam['w2c']for cam in  data['novel_view']], dim=0).cuda()
    Ks = torch.cat([cam['K']for cam in  data['novel_view']], dim=0).cuda()
    H, W = data['H'].item(), data['W'].item()
    renders, pred_alphas, info = rasterize_splats(splats , w2cs, Ks, W, H, render_mode="RGB+D", sh_degree=3, rasterize_feat=False)
    pred_imgs, pred_depths = renders[..., 0:3], renders[..., 3:4]
    temp_depths = []
    for depth in pred_depths:
        mask = depth_filter_grad(depth.squeeze(-1), ratio=0.5, max_depth=None)
        depth = torch.where(mask, depth.squeeze(-1), torch.tensor(0.0))
        temp_depths.append(depth)
    temp_depths = torch.stack(temp_depths, dim=0)
    pred_depths = temp_depths.cpu().numpy()
    
    pred_imgs = pred_imgs.detach().cpu().numpy()
    pred_imgs = np.clip(pred_imgs * 255, 0, 255).astype(np.uint8)
    
    centroids = superpixel_data['centroids']
    
    all_pts_xyz = []
    all_pts_rgb = []
    all_pts_cnt_xyz = []
    all_pts_cnt_rgb = []
    all_uv_mask = []
    for cam_idx, (depth, img) in enumerate(zip(pred_depths, pred_imgs)):
        cnts = centroids[cam_idx]
         # diffuse depth 
        u = np.arange(0, W)
        v = np.arange(0, H)
        u, v = np.meshgrid(u, v)
        uv = np.stack([v, u], axis=-1).reshape(-1, 2)[masks_comb[cam_idx].reshape(-1)]
      
        selected_depth = depth[uv[:, 0].astype(np.int32), uv[:, 1].astype(np.int32)]
        pts3d_rgb = img[uv[:, 0].astype(np.int32), uv[:, 1].astype(np.int32)]
        K = Ks[cam_idx].cpu().numpy()
        x = (uv[:, 1] - K[0, 2]) / K[0, 0] 
        y = (uv[:, 0] - K[1, 2]) / K[1, 1]
        x = x.reshape(-1, 1) * selected_depth.reshape(-1, 1)
        y = y.reshape(-1, 1) * selected_depth.reshape(-1, 1)
        z = selected_depth.reshape(-1, 1)
        pts3d = np.concatenate([x, y, z], axis=-1) # (N, 3)
        c2w = c2ws[cam_idx].cpu().numpy()
        pts3d_world = c2w[:3, :3] @ pts3d.reshape(-1, 3).T + c2w[:3, 3:4]
        
        pts3d_cnt_rgb = img[cnts[:, 0].astype(np.int32), cnts[:, 1].astype(np.int32)]
        depth_cnt = depth[cnts[:, 0].astype(np.int32), cnts[:, 1].astype(np.int32)]
        x_cnt = (cnts[:, 1] - K[0, 2]) / K[0, 0] * depth_cnt 
        y_cnt = (cnts[:, 0] - K[1, 2]) / K[1, 1] * depth_cnt
        pts3d_cnt = np.stack([x_cnt, y_cnt, depth_cnt], axis=-1)
        pts3d_cnt = c2w[:3, :3] @ pts3d_cnt.T + c2w[:3, 3:4]

        final_uv_mask = uv.copy()
        pts3d_index_keep1 = None
        pts3d_index_keep2 = None
        
        pts3d_cnt_index_keep1 = None
        pts3d_cnt_index_keep2 = None
        
        if bg_filter:
            pts3d_index = np.arange(pts3d_world.shape[1])
            pts3d_index_del = []
            pts3d_cnt_index = np.arange(pts3d_cnt.shape[1])
            pts3d_cnt_index_del = []
            for cam_idz in range(num_cam):
                pts3d_index_del.append(bg_filter_fn(w2c=w2cs[cam_idz].cpu().numpy(),
                          K=Ks[cam_idz].cpu().numpy(),
                          pts3d_world=pts3d_world,
                          pts3d_index=pts3d_index, 
                          masks_comb=masks_comb[cam_idz],
                          H=H, W=W))
                pts3d_cnt_index_del.append(bg_filter_fn(w2c=w2cs[cam_idz].cpu().numpy(),
                            K=Ks[cam_idz].cpu().numpy(),
                            pts3d_world=pts3d_cnt,
                            pts3d_index=pts3d_cnt_index, 
                            masks_comb=masks_comb[cam_idz],
                            H=H, W=W))
            pts3d_index_del = np.concatenate(pts3d_index_del, axis=0)
            pts3d_index_del = np.unique(pts3d_index_del)
            pts3d_index_keep1 = np.setdiff1d(pts3d_index, pts3d_index_del)
            pts3d_world = pts3d_world[:, pts3d_index_keep1]
            pts3d_rgb = pts3d_rgb[pts3d_index_keep1]
            final_uv_mask = final_uv_mask[pts3d_index_keep1]
            
            pts3d_cnt_index_del = np.concatenate(pts3d_cnt_index_del, axis=0)
            pts3d_cnt_index_del = np.unique(pts3d_cnt_index_del)
            pts3d_cnt_index_keep1 = np.setdiff1d(pts3d_cnt_index, pts3d_cnt_index_del)
            pts3d_cnt = pts3d_cnt[:, pts3d_cnt_index_keep1]
            pts3d_cnt_rgb = pts3d_cnt_rgb[pts3d_cnt_index_keep1]
            
        if depth_filter:
            pts3d_index = np.arange(pts3d_world.shape[1])
            pts3d_index_keep = []
            
            pts3d_cnt_index = np.arange(pts3d_cnt.shape[1])
            pts3d_cnt_index_keep = []
            
            for cam_idy in range(num_cam):
                if cam_idy == cam_idx:
                    continue
                pts3d_index_keep.append(depth_filter_fn(w2c=w2cs[cam_idy].cpu().numpy(), 
                             K=Ks[cam_idy].cpu().numpy(), 
                             pts3d_world=pts3d_world, 
                             pts3d_index=pts3d_index, 
                             H=H, W=W, 
                             nearby_depth=pred_depths[cam_idy], 
                             depth_offset_threshold=depth_offset_threshold))
                pts3d_cnt_index_keep.append(depth_filter_fn(w2c=w2cs[cam_idy].cpu().numpy(),
                                K=Ks[cam_idy].cpu().numpy(),
                                pts3d_world=pts3d_cnt,
                                pts3d_index=pts3d_cnt_index, 
                                H=H, W=W, 
                                nearby_depth=pred_depths[cam_idy], 
                                depth_offset_threshold=depth_offset_threshold))
            
            pts3d_index_keep2 = np.unique(np.concatenate(pts3d_index_keep, axis=0))
            pts3d_world = pts3d_world[:, pts3d_index_keep2]
            pts3d_rgb = pts3d_rgb[pts3d_index_keep2]
            
            pts3d_cnt_index_keep2 = np.unique(np.concatenate(pts3d_cnt_index_keep, axis=0))
            pts3d_cnt = pts3d_cnt[:, pts3d_cnt_index_keep2]
            pts3d_cnt_rgb = pts3d_cnt_rgb[pts3d_cnt_index_keep2]
            
            final_uv_mask=final_uv_mask[pts3d_index_keep2]
        uv_mask_canvas = np.zeros((H, W), dtype=np.bool_)
        uv_mask_canvas[final_uv_mask[:, 0], final_uv_mask[:, 1]] = True
        
        all_uv_mask.append(uv_mask_canvas)
        all_pts_xyz.append(pts3d_world.T)
        all_pts_rgb.append(pts3d_rgb)
        
        all_pts_cnt_xyz.append(pts3d_cnt.T)
        all_pts_cnt_rgb.append(pts3d_cnt_rgb)
        
    all_uv_mask = np.stack(all_uv_mask, axis=0)
    all_pts_xyz = np.concatenate(all_pts_xyz, axis=0)
    all_pts_rgb = np.concatenate(all_pts_rgb, axis=0)
    all_pts_cnt_xyz = np.concatenate(all_pts_cnt_xyz, axis=0)
    all_pts_cnt_rgb = np.concatenate(all_pts_cnt_rgb, axis=0)
    
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(all_pts_xyz)
    pcd.colors = o3d.utility.Vector3dVector(all_pts_rgb / 255)
    o3d.io.write_point_cloud(os.path.join(os.path.dirname(ckpt_path), f'diffuse_gs_{round_name}.ply'), pcd)
    
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(all_pts_cnt_xyz)
    pcd.colors = o3d.utility.Vector3dVector(all_pts_cnt_rgb / 255)
    o3d.io.write_point_cloud(os.path.join(os.path.dirname(ckpt_path), f'diffuse_gs_cnt_{round_name}.ply'), pcd)
    
    all_depth_masks = all_uv_mask
    mask_img = np.concatenate(all_depth_masks, axis=1)
    np.save(os.path.join(os.path.dirname(ckpt_path), f'depth_{round_name}.npy'), pred_depths)
    np.save(os.path.join(os.path.dirname(ckpt_path), f'mask_{round_name}.npy'), all_depth_masks)
    cv2.imwrite(os.path.join(os.path.dirname(ckpt_path), f'mask_{round_name}.png'), mask_img.astype(np.uint8) * 255)
    
    pred_imgs = np.concatenate(pred_imgs, axis=1)
    cv2.imwrite(os.path.join(os.path.dirname(ckpt_path), f'rgb_{round_name}.png'), pred_imgs[..., ::-1].astype(np.uint8))

    pred_depths = np.concatenate(pred_depths, axis=1)
    plt.imshow(pred_depths)
    plt.axis('off')
    plt.savefig(os.path.join(os.path.dirname(ckpt_path), f'render_depth_{round_name}.png'), dpi=300, bbox_inches='tight', pad_inches=0)
    plt.close()

def main():
    # first round
    args_src = [
                '--root', '/home/jiahao/Downloads/data/wildtrack_data_gt',
                '--round', '1_2',
                '--n-segments', '30',
                '--start-with', '0',
                '--end-with', '1',#'-1',
                '--loss_bg_w', '1.0',
                '--loss_rgb_w', '1.0',
                # '--loss_local_depth_w', '5e-5',
                '--loss_global_depth_w', '1e-3',
                '--loss_local_depth_w', '0',
                # '--loss_global_depth_w', '0',
                '--loss_sparsity_w', '0',
                '--loss_quantization_w', '1.0',#'1.0',
                '--loss_rank_depth_w', '5e-2',#'1e-2',
                '--init_opacity', '0.1',
                '--epochs', '2000',
                '--sparsity_epoch', '500',
                '--pruning_epoch', '500',
                "--bg_filter", "--no-depth_filter",
                '--depth_constraint', 'variance',
                '--random_select', '5'
            ]
    
    parser = ArgumentParser("Per scene training using Gaussian Splatting", add_help=True)
    parser.add_argument("--root", type=str, required=True, help="path to image folder")
    parser.add_argument("--round", type=str, required=True, help="round name")
    parser.add_argument("--start-with", type=int, default=0, help="the index of the first image to start with")
    parser.add_argument("--end-with", type=int, default=-1, help="the index of the last image to end with")
    parser.add_argument("--n-segments", type=int, default=30, help="the number of superpixels for each person")
    parser.add_argument("--init_strategy", type=str, default='depth_prior', choices=['depth_prior', 'uniform_sampling'], help="the number of superpixels for each person")
    parser.add_argument("--loss_bg_w", type=float, default=1.0, help="the weight of background loss")
    parser.add_argument("--loss_rgb_w", type=float, default=1.0, help="the weight of rgb loss")
    parser.add_argument("--loss_local_depth_w", type=float, default=1e-5, help="the weight of local depth loss")
    parser.add_argument("--loss_global_depth_w", type=float, default=1e-5, help="the weight of global depth loss")
    parser.add_argument("--loss_sparsity_w", type=float, default=1e-2, help="the weight of sparsity loss")
    parser.add_argument("--loss_rank_depth_w", type=float, default=1, help="the weight of rank depth loss")
    parser.add_argument("--loss_quantization_w", type=float, default=1.0, help="the weight of quantization loss")
    parser.add_argument("--epochs", type=int, default=2000, help="the number of epochs for training")
    parser.add_argument("--sparsity_epoch", type=int, default=2000, help="the number of epochs for sparsity")
    parser.add_argument("--pruning_epoch", type=int, default=500, help="the number of epochs for pruning")
    parser.add_argument("--init_opacity", type=float, default=0.1, help="the initial opacity of the gaussian")
    
    # parser.add_argument("--bg_filter", type=bool, default=True, help="whether to filter out the points out of the mask")
    # parser.add_argument("--depth_filter", type=bool, default=False, help="whether to filter out the points out of the mask based on depth")
    
    parser.add_argument("--bg_filter", action='store_true', help="whether to filter out the points out of the mask")
    parser.add_argument("--no-bg_filter", action='store_false', dest='bg_filter', help="Disable the bg_filter")

    parser.add_argument("--depth_filter", action='store_true', help="whether to filter out the points out of the mask based on depth")
    parser.add_argument("--no-depth_filter", action='store_false', dest='depth_filter', help="Disable the depth_filter")

    
    parser.add_argument("--depth_constraint", type=str, default='variance', choices=['variance', 'cluster'], help="the constraint type of depth")
    parser.add_argument("--random_select", type=int, default=None, help="the number of random selected masks for local depth constraint")
    
    args = parser.parse_args(args_src) # FOR DEBUG
    # args = parser.parse_args()
    print(args)
    
    data_root = args.root
    n_segments = args.n_segments
    num_depth_bin = 2_000
    
    depth_offset_threshold = 100
    # init dataset and dataloader
    start_with = args.start_with # start to process from `start_with` th frame
    end_with = args.end_with #-1 # end to process at `end_with` th frame
    print(f'The start_with and end_with are: {start_with} and {end_with}')
    
    wildtrack_detect_data = frameDataset(WildtrackDetection('/home/jiahao/Downloads/data/Wildtrack'))
    
    wildtrack_data = Wildtrack(data_root, mask_type='people', mask_label_type='split', start_with=start_with, end_with=end_with)
    wildtrack_dataloader = DataLoader(wildtrack_data, batch_size=1, pin_memory=True, num_workers=4, drop_last=True)
    sp_dataset = SuperPixels(data_root, n_segments=n_segments, HW=wildtrack_data.image_feat_size, start_with=start_with, end_with=end_with)
    assert len(wildtrack_data) == len(sp_dataset), f'Length of wildtrack_data and sp_dataset are not equal: {len(wildtrack_data)} and {len(sp_dataset)}'

    print('Start to generate matching label...')
    for idx, data in enumerate(wildtrack_dataloader):
        frame_idx = data['frame_idx'][0]
        superpixel_data = sp_dataset[idx]
        cur_t = time.time()
        
   
        # if args.round == '1_1':
        #     if args.init_strategy == 'depth_prior':
        #         all_pts3d_xyz, all_pts3d_rgb, all_images = step01_GS_init_depth_prior(frame_idx, data_root, wildtrack_detect_data[idx], data, superpixel_data, rebuild=True, bg_filter=False)
        #     else:
        #         all_pts3d_xyz, all_pts3d_rgb, all_images = step01_GS_init_us(frame_idx, data_root, data, num_depth_bin, superpixel_data, round_name=args.round, mask_min_area=-1)
        # else:
        #     all_images = torch.from_numpy(np.stack(superpixel_data['image'], axis=0)).cuda()
        #     save_folder = os.path.join(data_root, 'depths', frame_idx)
        #     super_pixel_ply = os.path.join(save_folder, f'init_gs_{args.round}.ply')
        #     assert os.path.exists(super_pixel_ply), f'Fail to find {super_pixel_ply}'

        print(f'[Frame {frame_idx}] Step01: init GS takes {time.time() - cur_t} s')
        all_images = torch.from_numpy(np.stack(superpixel_data['image'], axis=0)).cuda()
        

        all_images = data['nonnorm_image'].squeeze(0).cuda()
        mask = data['mask'].permute(1, 0, 2, 3).cuda()
        mask = torch.repeat_interleave(mask, repeats=3, dim=1)
        all_images[mask == 0] = 0
        all_images = all_images.permute(0, 2, 3, 1).cuda() / 255.0
        save_folder = os.path.join(data_root, 'depths', frame_idx)
        # super_pixel_ply = os.path.join(save_folder, f'init_gs_{args.round}.ply')
        super_pixel_ply = os.path.join(save_folder, f'init_gs_1_1.ply')
        # super_pixel_ply = os.path.join(save_folder, f'diffuse_gs_1_2.ply')
        
        print('super_pixel_ply:', super_pixel_ply)
        assert os.path.exists(super_pixel_ply), f'Fail to find {super_pixel_ply}'
        save_folder = os.path.join(data_root, 'depths', frame_idx, 'gs_optimize')
        if not os.path.exists(save_folder):
            os.makedirs(save_folder, exist_ok=True)
        empty_cache()
        cur_t = time.time()
        splats, trained_gs_path = step02_train(args, frame_idx, data_root, data, all_images, round_name=args.round, remove_init_gs= True if args.round == '1_1' else False, dataset=wildtrack_data, save_folder=save_folder, super_pixel_ply=super_pixel_ply)
        print(f'[Frame {frame_idx}] Step02: train GS takes {time.time() - cur_t} s')
        
        save_folder = os.path.join(data_root, 'depths', frame_idx, 'gs_optimize')
        trained_gs_path = os.path.join(save_folder, f'gs_{args.round}.pth')
        empty_cache()
        cur_t = time.time()
        step03_generate_matching_label(trained_gs_path, data, superpixel_data, depth_offset_threshold=depth_offset_threshold, round_name=args.round, bg_filter=args.bg_filter, depth_filter=args.depth_filter)
        print(f'[Frame {frame_idx}] Step03: generate matching label takes {time.time() - cur_t} s')
        # break
        
if __name__ == '__main__':
    main()