import os, sys
sys.path.append(os.path.join(os.getcwd(), 'submodules/MvCHM'))

import time
import torch
import random
import argparse
import numpy as np
from torch import nn
import open3d as o3d
from tqdm import tqdm
import torch.optim as optim
from datetime import datetime
import matplotlib.pyplot as plt
from argparse import ArgumentParser
from torch.utils.data import DataLoader
from submodules.MvCHM.lib.utils.visual_utils import Process, Monitor
from submodules.MvCHM.lib.data.wildtrack import Wildtrack as MvCHMWildtrack
from submodules.MvCHM.lib.data.dataloader import MultiviewDataset as MvCHMMultiviewDataset
from submodules.MvCHM.model.loss.loss import compute_loss
from submodules.MvCHM.lib.utils.visual_utils import visualize_heatmap

from dataset.wildtrack import Wildtrack, SuperPixels, WildtrackDetection, frameDataset
from utils.evaluation.evaluate import evaluate_rcll_prec_moda_modp

from supervised_cluster import parse_config, encode_postion, PointPillar

def Evaluate(args_src , thresh = 0.51, eps=1e-5):
    from unsupervised_cluster import FormatPRData
    DetectDataPath = '/home/jiahao/Downloads/data/Wildtrack'
    GSDataPath = '/home/jiahao/Downloads/data/wildtrack_data_gt'
    dataset_val = MvCHMMultiviewDataset(MvCHMWildtrack(DetectDataPath), set_name='val')
    val_dataloader = DataLoader( dataset_val, num_workers=1, batch_size=1)
    
    args, cfg = parse_config(args_src)
    model = PointPillar(cfg, torch.device('cuda'))
    checkpoint = torch.load(args.sup_decoder_checkpoint, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    PR_pred = FormatPRData(args.pr_dir_pred)
    PR_gt = FormatPRData(args.pr_dir_gt)
    
    device = torch.device('cuda')
    monitor = Monitor()
    with tqdm(iterable=val_dataloader, desc=f'[EVALUATE] ', postfix=dict, mininterval=1) as pbar:
        for batch_idx, data in enumerate(val_dataloader):
    
            with torch.no_grad():
                clustergs_path = os.path.join(GSDataPath, 'depths', f"{data['frame_idx'].item():08d}", 'diffuse_gs_1_2.ply')
                print('clustergs_path: ', clustergs_path)
                pcd = o3d.io.read_point_cloud(clustergs_path)
                gs_xyz = np.array(pcd.points)
                gs_rgb = np.array(pcd.colors)
                gs_xyz = dataset_val.base.get_worldgrid_from_worldcoord_Tensor(gs_xyz)
                point_clouds = np.concatenate([gs_xyz, gs_rgb], axis=1)
                point_clouds = torch.from_numpy(point_clouds).float().to(device='cuda')
                data['point_clouds'] = point_clouds 
                batch_dict, batch_pred = model(data)
                heatmap = torch.Tensor(data['heatmap']).to(device)
                plt.figure(figsize=(20, 10))
                plt.subplot(121)
                plt.imshow(heatmap.squeeze().cpu().numpy()) 
                plt.subplot(122)
                plt.imshow(torch.sigmoid(batch_pred['heatmap']).squeeze().cpu().numpy())
                plt.savefig(f'experiments/2024-10-08_19-41-57_wt/heatmaps/heatmap{batch_idx}.jpg', dpi=300, bbox_inches='tight', pad_inches=0)
                plt.close()
                gt_pos = encode_postion(heatmap = heatmap, mode = 'gt', grid_reduce = dataset_val.base.grid_reduce)
                pred_pos = encode_postion(heatmap = batch_pred['heatmap'], mode = 'pred', grid_reduce = dataset_val.base.grid_reduce, thresh = thresh, nms=True) # (n, 3)
                print("gt: ", gt_pos.shape, " pred: ", pred_pos.shape)
                PR_pred.add_item(pred_pos, batch_idx)
                PR_gt.add_item(gt_pos, batch_idx)
                pbar.update(1)
        PR_pred.save()
        PR_gt.save()
    
    recall, precision, moda, modp = evaluate_rcll_prec_moda_modp(args.pr_dir_pred, args.pr_dir_gt)
    print(f'\nEvaluation: threshold: {thresh}, MODA {moda:.1f}, MODP {modp:.1f}, prec {precision:.1f}, rcll {recall:.1f}')
            
if __name__ == '__main__':
    '''
    # Evaluate 
    '''
    args_src = [
        '--pr_dir_pred', 'output/exp_sup/pr_dir_pred.txt',
        '--pr_dir_gt', 'output/exp_sup/pr_dir_gt.txt',
        '--sup_decoder_checkpoint', 'experiments/2024-10-08_19-41-57_wt/checkpoints/Epoch39_train_loss0.0280_val_loss1.7337.pth'
    ]
    Evaluate(args_src, thresh = 0.7)
   