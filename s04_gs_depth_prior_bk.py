import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import cv2
import glob
import torch
import tqdm
import time
import numpy as np
import open3d as o3d
from PIL import Image
import torch.nn.functional as F
import matplotlib.pyplot as plt
from transformers import pipeline
from dataset.wildtrack import Wildtrack, SuperPixels
from torch.utils.data import DataLoader
from ProteusLib import SuperPixelGaussian, render, depth_constraints, DisparityToDepth, rasterize_splats, init_splats_from_ply
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from argparse import ArgumentParser
from utils.argument_utils import OptimizationParams
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
                # if cam_idx == cam_idy:
                #     continue
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

def step01_GS_init_depth_prior(frame_idx, data_root, data, superpixel_data, depth_threshold=3, round_name='1_1', rebuild=True, bg_filter=True):
    save_root = os.path.join(data_root, 'depths')
    if not os.path.exists(save_root):
        os.makedirs(save_root)
    save_folder = os.path.join(save_root, frame_idx)
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    if not os.path.exists(os.path.join(save_folder, f'init_gs_{round_name}.ply')) or rebuild:
        print('Predicting depth using depth-anything/Depth-Anything-V2-Large-hf...')
        depth_model = pipeline(task="depth-estimation", model="depth-anything/Depth-Anything-V2-Large-hf", device=0)
        assert os.path.exists('depth_wt/scales.npy'), 'Fail to find depth_wt/scales.npy'
        scales = np.load('depth_wt/scales.npy')
        num_cam = len(scales)
        
        all_pts3d_xyz = []
        all_pts3d_rgb = []
        
        target_masks = data['mask'].squeeze(0)
        target_masks = torch.where(target_masks>0, torch.ones_like(target_masks), torch.zeros_like(target_masks)).to(torch.bool)
        for cam_idx in range(num_cam):
            image = data['nonnorm_image'].squeeze(0)[cam_idx].permute(1, 2, 0)
            imagePIL = Image.fromarray(image.cpu().numpy().astype(np.uint8))
            target_mask = target_masks[cam_idx]
            
            w, h = imagePIL.size
            disp = depth_model(imagePIL)['predicted_depth']
            disp = (disp - disp.min()) / (disp.max() - disp.min())
            disp = F.interpolate(disp[:, None], (h, w), mode="bilinear", align_corners=True)[0, 0]
            pred_depth, mask = DisparityToDepth()(disp, depth_min=0.01, depth_max=15)
            depth_grad_x = torch.gradient(pred_depth, dim=0)[0]
            depth_grad_y = torch.gradient(pred_depth, dim=1)[0]
            
            depth_grad = torch.stack((depth_grad_x, depth_grad_y), dim=-1)
            depth_grad = torch.norm(depth_grad, dim=-1)
            region_with_valid_depth_gradient = depth_grad <= torch.median(depth_grad*depth_threshold)
            # mask = torch.logical_and(mask, region_with_valid_depth_gradient)
            mask = torch.logical_and(torch.logical_and(mask, region_with_valid_depth_gradient), target_mask)
            z = torch.where(mask, pred_depth, torch.zeros_like(pred_depth))
            z *= scales[cam_idx]
            yy, xx = torch.meshgrid(torch.arange(h), torch.arange(w))
            K = data['novel_view'][cam_idx]['K'].squeeze(0)
            c2w = data['novel_view'][cam_idx]['c2w'].squeeze(0)
            x = ( xx - K[0, 2] ) * z / K[0, 0]
            y = ( yy - K[1, 2] ) * z / K[1, 1]
            # contrust point cloud
            point_cloud = torch.stack((x, y, z), axis=-1)
            point_cloud = point_cloud[mask].reshape(-1, 3)
            
            pts3d_rgb = image[mask].reshape(-1, 3)
            pts3d_world = c2w[:3, :3] @ point_cloud.T + c2w[:3, 3:]
            if bg_filter:
                for cam_idy in range(num_cam):
                    if cam_idx == cam_idy:
                        continue
                    pts3d_index = torch.arange(pts3d_world.shape[1], device=pts3d_world.device)
                    human_mask = target_masks[cam_idy]
                    w2c = data['novel_view'][cam_idy]['w2c'].squeeze(0)
                    K = data['novel_view'][cam_idy]['K'].squeeze(0)
                    pts3d_cam = w2c[:3, :3] @ pts3d_world + w2c[:3, 3:4]
                    pts3d_scn = K @ pts3d_cam
                    pts3d_scn = pts3d_scn[:2, :] / pts3d_scn[2, :]
                    pts3d_scn = pts3d_scn.to(dtype=torch.int32)
                    # if pts3d_scn's pixel in the mask and within the image, then it is valid
                    mask1 = (pts3d_scn[0, :] >= 0) & (pts3d_scn[0, :] < w) & (pts3d_scn[1, :] >= 0) & (pts3d_scn[1, :] < h)
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
        pcd.points = o3d.utility.Vector3dVector(all_pts3d_xyz)
        pcd.colors = o3d.utility.Vector3dVector(all_pts3d_rgb/255.0)
        o3d.io.write_point_cloud(os.path.join(save_folder, f'init_gs_{round_name}.ply'), pcd)
    else:
        pcd = o3d.io.read_point_cloud(os.path.join(save_folder, f'init_gs_{round_name}.ply'))
        all_pts3d_xyz = torch.from_numpy(np.asarray(pcd.points)).cuda()
        all_pts3d_rgb = torch.from_numpy(np.asarray(pcd.colors)).cuda()
        
    all_images = torch.from_numpy(np.stack(superpixel_data['image'], axis=0)).cuda()
    return all_pts3d_xyz, all_pts3d_rgb, all_images
def step02_train(
         frame_idx,
         data_root, 
         data,
         all_images,
         epochs=1000,
         pruning_epoch=500,
         round_name='1_1',
         remove_init_gs=True,
         dataset=None    
         ):
    
    # gt_mask = data['mask'].permute(1, 0, 2, 3)
    masks = data['mask'].squeeze(0).cuda()
    gt_mask_comb_b = torch.where(masks > 0, torch.ones_like(masks), torch.zeros_like(masks)) # (7, h, w)
    gt_mask_split = masks
    gt_mask_comb_f = gt_mask_comb_b.float()
    gt_mask_comb_b = gt_mask_comb_b.bool().unsqueeze(-1)
    
    # init super pixel data
    gt_imgs = all_images.to(dtype=torch.float32)
    
    # init arguments
    parser = ArgumentParser(description="Training script parameters")
    op = OptimizationParams(parser)
    
    args = parser.parse_args([])
    opt = op.extract(args)
    # init model
    save_folder = os.path.join(data_root, 'depths', frame_idx)
    super_pixel_ply = os.path.join(save_folder, f'init_gs_{round_name}.ply')
    assert os.path.exists(super_pixel_ply), f'Fail to find {super_pixel_ply}'
    splats, optimizers = init_splats_from_ply(super_pixel_ply, scene_scale=dataset.scene_scale)
    strategy = DefaultStrategy(refine_start_iter=100, refine_stop_iter=1900, reset_every=1000, refine_every=1000, verbose=True)
    strategy.check_sanity(splats, optimizers)
    strategy_state = strategy.initialize_state(scene_scale=dataset.scene_scale)
    schedulers = [
        # means has a learning rate schedule, that end at 0.01 of the initial value
        torch.optim.lr_scheduler.ExponentialLR(
            optimizers["means"], gamma=0.01 ** (1.0 / epochs)
        ),
    ]
    ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to('cuda')
    # hyperparameters
    # max_opacity = 0.1
    loss_bg_w = 1       
    loss_rgb_w = 1
    
    loss_local_depth_w = 1e-5
    # loss_sparsity_w = 1e-2
    loss_quantization_w = 1
    # loss_global_depth_w = 0.1
    
    pbar = tqdm.tqdm(range(epochs))
    for epoch in pbar:
        
        losses = []
        w2cs = torch.cat([cam['w2c'] for cam in data['novel_view']], dim=0).float().cuda()
        Ks = torch.cat([cam['K'] for cam in data['novel_view']], dim=0).float().cuda()
        width = data['novel_view'][0]['width'].item()
        height = data['novel_view'][0]['height'].item()
        
        renders, alphas, info = rasterize_splats(splats, w2cs, Ks, width, height)
        colors, depths = renders[..., 0:3], renders[..., 3:4]
            
        strategy.step_pre_backward(
                params=splats,
                optimizers=optimizers,
                state=strategy_state,
                step=epoch,
                info=info,
            )
        
        # global depth constraints
        # loss_global_depth = pearson_correlation_loss(pred_metric_depth, lm_rel_depth, gt_mask_comb_b)
        
        # local depth constraints
        loss_local_depth = depth_constraints(depths, gt_mask_split, data['novel_view'], loss_type='cluster')
        
        # mse loss
        l1loss = F.l1_loss(colors, gt_imgs)
        ssimloss = 1.0 - ssim(
            gt_imgs.permute(0, 3, 1, 2), colors.permute(0, 3, 1, 2)
        )
        loss_rgb = 0.8 * l1loss + 0.2 * ssimloss
        
        # bg loss
        loss_bg = torch.nn.functional.l1_loss(alphas.squeeze(-1), gt_mask_comb_f)
        
        #quantization loss
        pred_opacity = torch.sigmoid(splats["opacities"])
        loss_quantization = torch.mean(pred_opacity * (1 - pred_opacity))
        
        loss = loss_rgb * loss_rgb_w + loss_bg * loss_bg_w + loss_local_depth * loss_local_depth_w + loss_quantization * loss_quantization_w #+ loss_global_depth * loss_global_depth_w
        
        losses.append(loss.detach().cpu().numpy())
        loss.backward()
        
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
            
        pbar.set_description(f'Epoch {epoch} Loss: {np.mean(losses):.6f}')
        pbar.update()
        
    data = {"splats": splats.state_dict()}
    torch.save(data, os.path.join(save_folder, f'gs_{round_name}.pth'))
    
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(splats['means'].detach().cpu().numpy())
    o3d.io.write_point_cloud(os.path.join(save_folder, f'gs_{round_name}.ply'), pcd)
    return splats, os.path.join(save_folder, f'gs_{round_name}.pth')

def step03_generate_matching_label(ckpt_path, data, num_cam = 7, depth_offset_threshold=100, bg_filter=True, depth_filter=True, round_name='1_1'):
    masks_split = data['mask'].squeeze(0)
    masks_comb = torch.where(masks_split>0, torch.ones_like(masks_split), torch.zeros_like(masks_split)).cpu().numpy()
    masks_comb = masks_comb.astype(np.bool_)
    c2ws = torch.cat([cam['c2w']for cam in  data['novel_view']], dim=0)
    w2cs = torch.cat([cam['w2c']for cam in  data['novel_view']], dim=0)
    Ks = torch.cat([cam['K']for cam in  data['novel_view']], dim=0)
    H, W = data['H'].item(), data['W'].item()
    
    # model = SuperPixelGaussian(None, max_sh_degree=3, required_grad=False)
    # model.load_from_ckpt(ckpt_path)
    
    # with torch.no_grad():   
    #     pred_imgs, pred_depths, pred_alphas = render(model, data['novel_view'])
    splats = torch.load(ckpt_path)['splats']
    
    renders, alphas, info = rasterize_splats(splats, w2cs.cuda(), Ks.cuda(), W, H)
    pred_depths = renders[..., 3:4]
    for depth in pred_depths:
        plt.imshow(depth.squeeze().detach().cpu().numpy())
        plt.show()
    pred_imgs = renders[..., 0:3]
    pred_depths = pred_depths.detach().cpu().numpy().squeeze(-1)

    pred_imgs = pred_imgs.detach().cpu().numpy()
    
    all_pts_xyz = []
    all_pts_rgb = []
    all_uv_mask = []
    for cam_idx, (depth, img) in enumerate(zip(pred_depths, pred_imgs)):
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

        final_uv_mask = uv.copy()
        pts3d_index_keep1 = None
        pts3d_index_keep2 = None
        if bg_filter:
            pts3d_index = np.arange(pts3d_world.shape[1])
            pts3d_index_del = []
            for cam_idz in range(num_cam):
                w2c = w2cs[cam_idz].cpu().numpy()
                K = Ks[cam_idz].cpu().numpy()
                pts3d_cam = w2c[:3, :3] @ pts3d_world + w2c[:3, 3:4]
                pts3d_scrn = K @ pts3d_cam
                pts3d_scrn = pts3d_scrn[:2, :] / pts3d_scrn[2:3, :]
                mask1 = (pts3d_scrn[0, :] > 0) & (pts3d_scrn[0, :] < W) & (pts3d_scrn[1, :] > 0) & (pts3d_scrn[1, :] < H)
                pts3d_scrn = pts3d_scrn[:, mask1]
                mask2 = masks_comb[cam_idz]
                mask2 = mask2[pts3d_scrn[1].astype(np.int32), pts3d_scrn[0].astype(np.int32)]
                pts3d_index_del.append(pts3d_index[mask1][~mask2])
            pts3d_index_del = np.concatenate(pts3d_index_del, axis=0)
            pts3d_index_del = np.unique(pts3d_index_del)
            pts3d_index_keep1 = np.setdiff1d(pts3d_index, pts3d_index_del)
            pts3d_world = pts3d_world[:, pts3d_index_keep1]
            pts3d_rgb = pts3d_rgb[pts3d_index_keep1]
            final_uv_mask = final_uv_mask[pts3d_index_keep1]
        if depth_filter:
            pts3d_index = np.arange(pts3d_world.shape[1])
            pts3d_index_keep = []
            for cam_idy in range(num_cam):
                if cam_idy == cam_idx:
                    continue
                w2c = w2cs[cam_idy].cpu().numpy()
                K = Ks[cam_idy].cpu().numpy()
                pts3d_cam = w2c[:3, :3] @ pts3d_world + w2c[:3, 3:4]
                render_depth = pts3d_cam[-1, :]
                pts3d_scrn = K @ pts3d_cam
                pts3d_scrn = pts3d_scrn[:2, :] / pts3d_scrn[2:3, :]
                mask1 = (pts3d_scrn[0, :] > 0) & (pts3d_scrn[0, :] < W) & (pts3d_scrn[1, :] > 0) & (pts3d_scrn[1, :] < H)
                pts3d_scrn = pts3d_scrn[:, mask1]
                nearby_depth = pred_depths[cam_idy]
                nearby_depth = nearby_depth[pts3d_scrn[1].astype(np.int32), pts3d_scrn[0].astype(np.int32)]
                depth_dist = np.abs(nearby_depth - render_depth[mask1])
                mask2 = depth_dist < depth_offset_threshold
                pts3d_index_keep.append(pts3d_index[mask1][mask2])
            
            pts3d_index_keep2 = np.unique(np.concatenate(pts3d_index_keep, axis=0))
            pts3d_world = pts3d_world[:, pts3d_index_keep2]
            pts3d_rgb = pts3d_rgb[pts3d_index_keep2]
            
            final_uv_mask=final_uv_mask[pts3d_index_keep2]
        uv_mask_canvas = np.zeros((H, W), dtype=np.bool_)
        uv_mask_canvas[final_uv_mask[:, 0], final_uv_mask[:, 1]] = True
        
        all_uv_mask.append(uv_mask_canvas)
        all_pts_xyz.append(pts3d_world.T)
        all_pts_rgb.append(pts3d_rgb)
        
    all_uv_mask = np.stack(all_uv_mask, axis=0)
    all_pts_xyz = np.concatenate(all_pts_xyz, axis=0)
    all_pts_rgb = np.concatenate(all_pts_rgb, axis=0)
    
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(all_pts_xyz)
    pcd.colors = o3d.utility.Vector3dVector(all_pts_rgb / 255)
    o3d.io.write_point_cloud(os.path.join(os.path.dirname(ckpt_path), f'diffuse_gs_{round_name}.ply'), pcd)
    
    all_depth_masks = all_uv_mask
    mask_img = np.concatenate(all_depth_masks, axis=1)
    np.save(os.path.join(os.path.dirname(ckpt_path), f'depth_{round_name}.npy'), pred_depths)
    np.save(os.path.join(os.path.dirname(ckpt_path), f'mask_{round_name}.npy'), all_depth_masks)
    cv2.imwrite(os.path.join(os.path.dirname(ckpt_path), f'mask_{round_name}.png'), mask_img.astype(np.uint8) * 255)


def main():
    args_src = ['--root', '/home/jiahao/Downloads/data/wildtrack_data', '--round', '1_1', '--n-segments', '30', '--start-with', '0', '--end-with', '1']
    parser = ArgumentParser("Per scene training using Gaussian Splatting", add_help=True)
    parser.add_argument("--root", type=str, required=True, help="path to image folder")
    parser.add_argument("--round", type=str, required=True, help="round name")
    parser.add_argument("--start-with", type=int, default=0, help="the index of the first image to start with")
    parser.add_argument("--end-with", type=int, default=-1, help="the index of the last image to end with")
    parser.add_argument("--n-segments", type=int, default=30, help="the number of superpixels for each person")
    parser.add_argument("--init_strategy", type=str, default='depth_prior', choices=['depth_prior', 'uniform_sampling'], help="the number of superpixels for each person")
    args = parser.parse_args(args_src) # FOR DEBUG
    # args = parser.parse_args()
    print(args)
    
    data_root = args.root
    n_segments = args.n_segments
    num_depth_bin = 2_000
    epochs = 2000
    depth_offset_threshold = 100
    # init dataset and dataloader
    start_with = args.start_with # start to process from `start_with` th frame
    end_with = args.end_with #-1 # end to process at `end_with` th frame
    print(f'The start_with and end_with are: {start_with} and {end_with}')
    
    wildtrack_data = Wildtrack(data_root, mask_type='people', mask_label_type='split', start_with=start_with, end_with=end_with)
    wildtrack_dataloader = DataLoader(wildtrack_data, batch_size=1, pin_memory=True, num_workers=4, drop_last=True)
    sp_dataset = SuperPixels(data_root, n_segments=n_segments, HW=wildtrack_data.image_feat_size, start_with=start_with, end_with=end_with)
    assert len(wildtrack_data) == len(sp_dataset), f'Length of wildtrack_data and sp_dataset are not equal: {len(wildtrack_data)} and {len(sp_dataset)}'

    print('Start to generate matching label...')
    for idx, data in enumerate(wildtrack_dataloader):
        frame_idx = data['frame_idx'][0]
        superpixel_data = sp_dataset[idx]
        cur_t = time.time()
        if args.init_strategy == 'depth_prior':
            all_pts3d_xyz, all_pts3d_rgb, all_images = step01_GS_init_depth_prior(frame_idx, data_root, data, superpixel_data, rebuild=False, bg_filter=False)
        else:
            all_pts3d_xyz, all_pts3d_rgb, all_images = step01_GS_init_us(frame_idx, data_root, data, num_depth_bin, superpixel_data, round_name=args.round, mask_min_area=-1)
        print(f'[Frame {frame_idx}] Step01: init GS takes {time.time() - cur_t} s')
        
        empty_cache()
        cur_t = time.time()
        model, trained_gs_path = step02_train(frame_idx, data_root, data, all_images, epochs, round_name=args.round, dataset=wildtrack_data)
        print(f'[Frame {frame_idx}] Step02: train GS takes {time.time() - cur_t} s')
        
        # trained_gs_path = '/home/jiahao/Downloads/data/wildtrack_data/depths/00000000/gs_1_1.pth'
        empty_cache()
        cur_t = time.time()
        step03_generate_matching_label(trained_gs_path, data, depth_offset_threshold=depth_offset_threshold, round_name=args.round, bg_filter=True, depth_filter=True)
        print(f'[Frame {frame_idx}] Step03: generate matching label takes {time.time() - cur_t} s')
        break
        
if __name__ == '__main__':
    main()