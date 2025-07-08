'''
    Fine-tune the monocular depth estimation model on the wildtrack dataset
'''
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import cv2
import time
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


from dataset.wildtrack import WildtrackDepthEstimation
from submodules.depth_anything_v2.dpt import DepthAnythingV2
from submodules.depth_anything_v2.metric_depth.util.loss import SiLogLoss
from submodules.depth_anything_v2.metric_depth.util.metric import eval_depth
from submodules.depth_anything_v2.metric_depth.util.utils import init_log
from ProteusLib import depth_constraints
from dataset.wildtrack import Wildtrack, SuperPixels
from s04_gs_us import bg_filter_fn, depth_filter_fn

parser = argparse.ArgumentParser(description='Depth Anything V2 for Metric Depth Estimation')

parser.add_argument('--encoder', default='vitl', choices=['vits', 'vitb', 'vitl', 'vitg'])
parser.add_argument('--dataset', default='wildtrack', choices=['hypersim', 'vkitti', 'wildtrack'])
parser.add_argument('--img-size', default=518, type=int)
parser.add_argument('--min-depth', default=0.001, type=float)
parser.add_argument('--max-depth', default=20, type=float)
parser.add_argument('--epochs', default=40, type=int)
parser.add_argument('--bs', default=2, type=int)
parser.add_argument('--lr', default=0.000005, type=float)
parser.add_argument('--pretrained-from', type=str)
parser.add_argument('--save-path', type=str, required=True)
parser.add_argument('--local-rank', default=0, type=int)
parser.add_argument('--port', default=0, type=int)
parser.add_argument('--data_root', default="/home/jiahao/Downloads/data/wildtrack_data", type=str)
parser.add_argument('--loss_depth_w', default=1, type=float)
parser.add_argument('--loss_depth_consistency_w', default=0.05, type=float)
parser.add_argument('--round', default='1_1', type=str)
parser.add_argument("--n-segments", type=int, default=30, help="the number of superpixels for each person")
parser.add_argument("--start-with", type=int, default=0, help="the index of the first image to start with")
parser.add_argument("--end-with", type=int, default=-1, help="the index of the last image to end with")
parser.add_argument("--depth_mask_type", type=str, default='pseudo_label', choices=['matching_label', 'sam_label', 'pseudo_label'], help="the index of the last image to end with")

def cm2m(x):
    return x / 100

def m2cm(x):
    return x * 100

def consistent_depth_constraint(depth, split_mask, random_select=5, least_mask_num=50):
    losses = 0
    for cam_idx, sm in enumerate(split_mask):
        for sm_idx in torch.unique(sm):
            if sm_idx == 0:
                continue # skip the background
            selected_depth = depth[cam_idx, sm==sm_idx]
            num_mask_total = len(selected_depth)
            if num_mask_total<least_mask_num:
                continue
            num_mask_select = min(num_mask_total, random_select)
            # randomly generate `num_mask_select` index 
            mask_idx = torch.randperm(num_mask_total)[:num_mask_select]
            if mask_idx.shape[0] == 1:
                continue
            losses += torch.var(selected_depth[mask_idx])
    return losses


def main():
    args_src = [
              "--epochs", "20",
              "--encoder", "vits",
              "--bs", "1",
              "--lr", "0.000005",
              "--save-path", "output/1_1_w_dc_randomselect5",#"output/1_1",
              "--dataset", "wildtrack",
              "--img-size", "518",
              "--min-depth", "0.001",
              "--max-depth", "40",
              "--pretrained-from", "checkpoints/depth_anything_v2_vits.pth",
              "--port", "20596",
              "--data_root", "/home/jiahao/Downloads/data/wildtrack_data_gt",
              "--round", "1_1",
              '--depth_mask_type', 'pseudo_label',
            ]
    # args = parser.parse_args(args_src) # ROR DEBUG
    args = parser.parse_args() 
    
    warnings.simplefilter('ignore', np.RankWarning)
    
    logger = init_log('global', logging.INFO)
    logger.propagate = 0
    
    rank, world_size = 0, 1
    local_rank = 0
    if rank == 0:
        all_args = {**vars(args), 'ngpus': world_size}
        logger.info('{}\n'.format(pprint.pformat(all_args)))
        writer = SummaryWriter(args.save_path)
    
    cudnn.enabled = True
    cudnn.benchmark = True
    
    # size = (args.img_size, args.img_size)
    ori_size = (540, 960)
    new_size = (518, 924) 
    size = new_size
    if args.dataset == 'wildtrack':
        trainset = WildtrackDepthEstimation(args.data_root, mode='train', size=size, depth_mask_type=args.depth_mask_type, round_name=args.round, data_ratio=0.9)
    else:
        raise NotImplementedError
    
    trainsampler = None
    trainloader = DataLoader(trainset, batch_size=args.bs, pin_memory=True, num_workers=4, drop_last=True, sampler=trainsampler)
    
    if args.dataset == 'wildtrack':
        valset = WildtrackDepthEstimation(args.data_root, mode='val', size=size,  depth_mask_type=args.depth_mask_type, round_name=args.round, data_ratio=0.9)
    else:
        raise NotImplementedError
    
    valsampler = None
    valloader = DataLoader(valset, batch_size=1, pin_memory=True, num_workers=4, drop_last=True, sampler=valsampler)
    
    model_configs = {
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
        'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
    }
    model = DepthAnythingV2(**{**model_configs[args.encoder], 'max_depth': args.max_depth})
    
    if args.pretrained_from:
        if 'depth_anything_v2' in args.pretrained_from:
            model.load_state_dict({k: v for k, v in torch.load(args.pretrained_from, map_location='cpu').items() if 'pretrained' in k}, strict=False)
        else:
            model.load_state_dict(torch.load(args.pretrained_from, map_location='cpu')['model'])
    
    # model.pretrained.requires_grad_(False)  # freeze the DINOv2 backbone
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model.cuda(local_rank)
    
    criterion = SiLogLoss().cuda(local_rank)
    
    optimizer = AdamW([{'params': [param for name, param in model.named_parameters() if 'pretrained' in name], 'lr': args.lr},
                       {'params': [param for name, param in model.named_parameters() if 'pretrained' not in name], 'lr': args.lr * 10.0}],
                      lr=args.lr, betas=(0.9, 0.999), weight_decay=0.01)
    
    # optimizer = AdamW(model.depth_head.parameters(), lr=args.lr, betas=(0.9, 0.999), weight_decay=0.01)
    total_iters = args.epochs * len(trainloader)
    
    previous_best = {'d1': 0, 'd2': 0, 'd3': 0, 'abs_rel': 100, 'sq_rel': 100, 'rmse': 100, 'rmse_log': 100, 'log10': 100, 'silog': 100}
    
    for epoch in range(args.epochs):
        if rank == 0:
            logger.info('===========> Epoch: {:}/{:}, d1: {:.3f}, d2: {:.3f}, d3: {:.3f}'.format(epoch, args.epochs, previous_best['d1'], previous_best['d2'], previous_best['d3']))
            logger.info('===========> Epoch: {:}/{:}, abs_rel: {:.3f}, sq_rel: {:.3f}, rmse: {:.3f}, rmse_log: {:.3f}, '
                        'log10: {:.3f}, silog: {:.3f}'.format(
                            epoch, args.epochs, previous_best['abs_rel'], previous_best['sq_rel'], previous_best['rmse'], 
                            previous_best['rmse_log'], previous_best['log10'], previous_best['silog']))
        
        # trainloader.sampler.set_epoch(epoch + 1)
        
        model.train()
        total_loss = 0
        
        for i, sample in enumerate(trainloader):
            optimizer.zero_grad()
            
            img, depth, valid_mask = sample['image'].cuda(), sample['depth'].cuda(), sample['valid_mask'].cuda()
            depth = cm2m(depth)

            img = img.squeeze(0)
            depth = depth.squeeze(0)
            valid_mask = valid_mask.squeeze(0).float()
            pred = model(img)
            # split_mask = sample['split_mask'].squeeze(0).cuda() 
            # loss_depth_consistency = consistent_depth_constraint(pred, split_mask)
            # loss_depth_consistency = depth_constraints(pred, split_mask, sample['novel_view'], loss_type='variance', random_select=5)
            loss_depth = criterion(pred, depth, (valid_mask == 1))
            loss = loss_depth * args.loss_depth_w #+ loss_depth_consistency * args.loss_depth_consistency_w
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            iters = epoch * len(trainloader) + i
            
            lr = args.lr * (1 - iters / total_iters) ** 0.9
            
            optimizer.param_groups[0]["lr"] = lr
            optimizer.param_groups[1]["lr"] = lr * 10.0
            
            if rank == 0:
                writer.add_scalar('train/loss', loss.item(), iters)
            
            if rank == 0 and i % 20 == 0:
                # logger.info('Iter: {}/{}, LR: {:.7f}, Loss: {:.3f}, Depth Loss: {:.3f}, DepthConstraint Loss: {:.3f}'.format(i, len(trainloader), optimizer.param_groups[0]['lr'], loss.item(), loss_depth.item(), loss_depth_consistency.item() * args.loss_depth_consistency_w ))
                logger.info('Iter: {}/{}, LR: {:.7f}, Loss: {:.3f}, Depth Loss: {:.3f}'.format(i, len(trainloader), optimizer.param_groups[0]['lr'], loss.item(), loss_depth.item()))
        
        model.eval()
        
        results = {'d1': torch.tensor([0.0]).cuda(), 'd2': torch.tensor([0.0]).cuda(), 'd3': torch.tensor([0.0]).cuda(), 
                   'abs_rel': torch.tensor([0.0]).cuda(), 'sq_rel': torch.tensor([0.0]).cuda(), 'rmse': torch.tensor([0.0]).cuda(), 
                   'rmse_log': torch.tensor([0.0]).cuda(), 'log10': torch.tensor([0.0]).cuda(), 'silog': torch.tensor([0.0]).cuda()}
        nsamples = torch.tensor([0.0]).cuda()
        
        for i, sample in enumerate(valloader):
            
            img, depth, valid_mask = sample['image'].cuda().float(), sample['depth'].cuda()[0], sample['valid_mask'].cuda()[0]
            depth = cm2m(depth)
            img = img.squeeze(0)
            depth = depth.squeeze(0)
            valid_mask = valid_mask.squeeze(0).float()
            
            with torch.no_grad():
                pred = model(img)
                # pred = F.interpolate(pred[:, None], depth.shape[-2:], mode='bilinear', align_corners=True)[0, 0]
            
            valid_mask = (valid_mask == 1)# & (depth >= args.min_depth) & (depth <= args.max_depth)
            
            if valid_mask.sum() < 10:
                continue
            
            cur_results = eval_depth(pred[valid_mask], depth[valid_mask])
            
            for k in results.keys():
                results[k] += cur_results[k]
            nsamples += 1
            if i > 20:
                break # only eval first 20 images
        if rank == 0:
            logger.info('==========================================================================================')
            logger.info('{:>8}, {:>8}, {:>8}, {:>8}, {:>8}, {:>8}, {:>8}, {:>8}, {:>8}'.format(*tuple(results.keys())))
            logger.info('{:8.3f}, {:8.3f}, {:8.3f}, {:8.3f}, {:8.3f}, {:8.3f}, {:8.3f}, {:8.3f}, {:8.3f}'.format(*tuple([(v / nsamples).item() for v in results.values()])))
            logger.info('==========================================================================================')
            print()
            
            for name, metric in results.items():
                writer.add_scalar(f'eval/{name}', (metric / nsamples).item(), epoch)
        
        for k in results.keys():
            if k in ['d1', 'd2', 'd3']:
                previous_best[k] = max(previous_best[k], (results[k] / nsamples).item())
            else:
                previous_best[k] = min(previous_best[k], (results[k] / nsamples).item())
        
        if rank == 0:
            checkpoint = {
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch,
                'previous_best': previous_best,
            }
            torch.save(checkpoint, os.path.join(args.save_path, 'latest.pth'))
            if epoch % 20 == 0:
                torch.save(checkpoint, os.path.join(args.save_path, f'{epoch:3d}.pth'))
                
    return os.path.join(args.save_path, 'latest.pth')

def vis_diff(ckpt_path):
    args_src = [
              "--epochs", "50",
              "--encoder", "vits",
              "--bs", "1",
              "--lr", "0.000005",#"0.000005",
              "--save-path", "output/1_1_wo_dc",
              "--dataset", "wildtrack",
              "--img-size", "518",
              "--min-depth", "0.001",
              "--max-depth", "40",
              "--pretrained-from", "output/1_1_wo_dc/latest.pth",
              "--port", "20596",
              "--data_root", "/home/jiahao/Downloads/data/wildtrack_data_gt",
              "--round", "1_1",
              '--depth_mask_type', 'pseudo_label',
            ]
    args = parser.parse_args(args_src) # ROR DEBUG
    # args = parser.parse_args() 
    
    round_th, stage_th = args.round.split('_')
    
    bg_filter = True
    depth_filter = False
    depth_offset_threshold = 100
    
    ori_size = (540, 960) # depth image size
    new_size = (518, 924) 
    local_rank = 0
    dataset = WildtrackDepthEstimation(args.data_root, mode='val', size=new_size, depth_mask_type=args.depth_mask_type, mask_choice='foreground', round_name=args.round,)#data_ratio=0.9)
    dataset_os = WildtrackDepthEstimation(args.data_root, mode='val', size=ori_size, depth_mask_type=args.depth_mask_type, mask_choice='foreground')
    w2cs = dataset_os.w2cs
    Ks = dataset_os.intrinsic_matrices
    del dataset_os
    
    num_cam = dataset.num_cam
    H, W = ori_size 
    sp_dataset = SuperPixels(args.data_root, n_segments=args.n_segments, HW=dataset.image_feat_size, start_with=args.start_with, end_with=args.end_with)
    
    trainsampler = None
    valloader = DataLoader(dataset, batch_size=args.bs, pin_memory=True, num_workers=4, drop_last=True, sampler=trainsampler)
    
    model_configs = {
        'vits': {'encoder': 'vits', 'features': 64,  'out_channels': [48, 96, 192, 384]},
        'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
        'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
    }
    model = DepthAnythingV2(**{**model_configs[args.encoder], 'max_depth': args.max_depth})
    model.load_state_dict(torch.load(ckpt_path, map_location='cpu', weights_only=True)['model'])
    model.cuda(local_rank)
    model.eval()
    for i, sample in enumerate(valloader):
        superpixel_data = sp_dataset[i]
        centroids = superpixel_data['centroids']
        img, masks_comb = sample['image'].cuda().float(), sample['valid_mask'].cuda()[0]
        img = img.squeeze(0)
        masks_comb = masks_comb.squeeze(0).float()

        with torch.no_grad():
            pred_depths = model(img)

        # renormalize
        img = img.permute(0, 2, 3, 1)
        img = (img.cpu() * dataset.std.reshape(1, 1, 1, 3) + dataset.mean.reshape(1, 1, 1, 3))
        img = torch.clamp(img, 0, 1)
        pred_depths = m2cm(pred_depths.cpu().numpy())
        pred_depths = [cv2.resize(p, (ori_size[1], ori_size[0]), interpolation=cv2.INTER_NEAREST) for p in pred_depths]
        masks_comb = [cv2.resize(vm.cpu().numpy(), (ori_size[1], ori_size[0]), interpolation=cv2.INTER_NEAREST).astype(np.bool_) for vm in masks_comb] 
        images = [cv2.resize(m.cpu().numpy(), (ori_size[1], ori_size[0]), interpolation=cv2.INTER_NEAREST) for m in img]
        # plt.subplot(121)
        # plt.imshow(pred_depths[0])
        # plt.subplot(122)
        # plt.imshow(masks_comb[0])
        # plt.show()
        all_pts_xyz = []
        all_pts_rgb = []
        all_pts_cnt_xyz = []
        all_pts_cnt_rgb = []
        all_render_masks = []
        
        for cam_idx in range(dataset.num_cam):
            
            mask = masks_comb[cam_idx]
            
            v, u = np.where(mask)
            uv = np.stack([v, u], axis=1)
            K = Ks[cam_idx]
            depth = pred_depths[cam_idx][v, u]
            pts3d_rgb = images[cam_idx][v, u]
            fx, cx, fy, cy = K[0, 0], K[0, 2], K[1, 1], K[1, 2]
            dx = (u - cx) / fx
            dy = (v - cy) / fy
            dir_cam = np.stack([dx, dy, np.ones_like(dx)], axis=1)
            
            dir_cam = dir_cam.reshape(-1, 3)
            pts_cam = dir_cam * depth.reshape(-1, 1)
            
            w2c44 = w2cs[cam_idx]
            c2w44 = np.linalg.inv(w2c44)
            pts3d_world = c2w44[:3, :3] @ pts_cam.T + c2w44[:3, 3:]
            
            cnts = centroids[cam_idx]
            pts3d_cnt_rgb = images[cam_idx][cnts[:, 0].astype(np.int32), cnts[:, 1].astype(np.int32)]
            depth_cnt = pred_depths[cam_idx][cnts[:, 0].astype(np.int32), cnts[:, 1].astype(np.int32)]
            x_cnt = (cnts[:, 1] - cx) / fx * depth_cnt 
            y_cnt = (cnts[:, 0] - cy) / fy * depth_cnt
            pts3d_cnt = np.stack([x_cnt, y_cnt, depth_cnt], axis=-1)
            pts3d_cnt = c2w44[:3, :3] @ pts3d_cnt.T + c2w44[:3, 3:4]
            
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
                    pts3d_index_del.append(bg_filter_fn(w2c=w2cs[cam_idz],
                            K=Ks[cam_idz],
                            pts3d_world=pts3d_world,
                            pts3d_index=pts3d_index, 
                            masks_comb=masks_comb[cam_idz],
                            H=H, W=W))
                    pts3d_cnt_index_del.append(bg_filter_fn(w2c=w2cs[cam_idz],
                                K=Ks[cam_idz],
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
                
            render_masks = np.zeros_like(mask)
            render_masks[final_uv_mask[:, 0], final_uv_mask[:, 1]] = 1
            all_render_masks.append(render_masks)
            all_pts_xyz.append(pts3d_world.T)
            all_pts_rgb.append(pts3d_rgb)

            all_pts_cnt_xyz.append(pts3d_cnt.T)
            all_pts_cnt_rgb.append(pts3d_cnt_rgb)
            
        all_pts_xyz = np.concatenate(all_pts_xyz, axis=0)
        all_pts_rgb = np.concatenate(all_pts_rgb, axis=0)
        all_pts_cnt_xyz = np.concatenate(all_pts_cnt_xyz, axis=0)
        all_pts_cnt_rgb = np.concatenate(all_pts_cnt_rgb, axis=0)
        
        all_render_masks = np.stack(all_render_masks, axis=0)
        all_render_masks_img = np.concatenate(all_render_masks, axis=1).astype(np.uint8) * 255
        save_folder = os.path.join(args.data_root, 'depths', sample['frame_idx'][0])
        # save_folder = os.path.dirname(ckpt_path)
        # save point cloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(all_pts_xyz)
        pcd.colors = o3d.utility.Vector3dVector(all_pts_rgb)
        o3d.io.write_point_cloud(os.path.join(save_folder, f'diffuse_gs_{round_th}_{int(stage_th)+1}.ply'), pcd)
        
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(all_pts_cnt_xyz)
        pcd.colors = o3d.utility.Vector3dVector(all_pts_cnt_rgb / 255)
        o3d.io.write_point_cloud(os.path.join(save_folder, f'diffuse_gs_cnt_{round_th}_{int(stage_th)+1}.ply'), pcd)
        
        # save mask npy
        np.save(os.path.join(save_folder, f'mask_{round_th}_{int(stage_th)+1}.npy'), all_render_masks)
        # save pred npy
        np.save(os.path.join(save_folder, f'depth_{round_th}_{int(stage_th)+1}.npy'), pred_depths)
        # save mask png
        cv2.imwrite(os.path.join(save_folder, f'mask_{round_th}_{int(stage_th)+1}.png'), all_render_masks_img)
        print(f'[{i}/{len(valloader)}] Finish saving {sample["frame_idx"][0]}.')
if __name__ == '__main__':
    # ckpt_path = main()
    # ckpt_path = 'output/final_wt/latest.pth' # ROR DEBUG
    ckpt_path = 'output/final_wt_full_training_data/180.pth' # ROR DEBUG
    vis_diff(ckpt_path)