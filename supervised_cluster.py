'''
The implementation of the supervised clustering algorithm is based on the following paper:
* Multiview Detection with Cardboard Human Modeling  [https://arxiv.org/abs/2207.02013]
* PointPillars: Fast Encoders for Object Detection from Point Clouds  [https://arxiv.org/abs/1812.05784]
'''
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
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from distutils.file_util import copy_file
from submodules.MvCHM.lib.utils.tool_utils import MetricDict, to_numpy
from submodules.MvCHM.lib.utils.visual_utils import Process, Monitor
from submodules.MvCHM.lib.utils.config_utils import cfg_from_yaml_file
from submodules.MvCHM.lib.utils.config_utils import cfg as mvchm_cfg
from submodules.MvCHM.model.head.head_pred import Head
from submodules.MvCHM.model.p2v.point2voxel import Point2Voxel
from submodules.MvCHM.model.pillar.pillar_vfe import PillarVFE
from submodules.MvCHM.model.m2b.bev_backbone import BaseBEVBackbone
from submodules.MvCHM.model.pillar.pillar_scatter import PointPillarScatter
from submodules.MvCHM.lib.data.wildtrack import Wildtrack as MvCHMWildtrack
from submodules.MvCHM.lib.data.dataloader import MultiviewDataset as MvCHMMultiviewDataset
from submodules.MvCHM.model.loss.loss import compute_loss
from submodules.MvCHM.lib.utils.visual_utils import visualize_heatmap

from dataset.wildtrack import Wildtrack, SuperPixels, WildtrackDetection, frameDataset
from utils.evaluation.evaluate import evaluate_rcll_prec_moda_modp
class PointPillar(nn.Module):
    def __init__(self, cfg, device:torch.device):
        super().__init__()
        
        self.device = device
        self.cfg = cfg

        # --- init Point2Voxel module --- #
        self.p2v = Point2Voxel(self.cfg).to(device=device)

        # ---- init PillarVFE module ---- #
        self.vfe = PillarVFE(self.cfg).to(device=device) # [n_pillars, 64]

        # ---- init PointPillarScatter module ---- #
        self.pps = PointPillarScatter(self.cfg).to(device=device) 

        # ---- init BaseBEVBackBone module ---- #
        self.bev = BaseBEVBackbone(self.cfg).to(device=device) 

        # ---- init Head module ---- #
        self.head = Head(self.cfg).to(device=device)

    def forward(self, batch_dict):
        # ---- convert points to pillar ---- #
        batch_dict = self.p2v(batch_dict)

        # ---- create pillar features ---- #
        batch_dict = self.vfe(batch_dict)

        # ---- create spatial features ---- #
        batch_dict = self.pps(batch_dict)

        # ---- create compact spatial features ---- #
        batch_dict = self.bev(batch_dict)

        # ---- prediction head ---- #
        batch_pred = self.head(batch_dict)
        
        return batch_dict, batch_pred
    
def Inference(args):
    print(args)
    data_root = args.root
    start_with = args.start_with # start to process from `start_with` th frame
    end_with = args.end_with #-1 # end to process at `end_with` th frame
    print(f'The start_with and end_with are: {start_with} and {end_with}')
    cfg_file = r'submodules/MvCHM/cfgs/MvDDE.yaml'
    cfg = cfg_from_yaml_file(cfg_file, mvchm_cfg)
    model = PointPillar(cfg, torch.device('cuda'))
    # wildtrack
    grid_range = [0, 0, 0, 480, 1440, 80]
    
    wildtrack_detect_data = frameDataset(WildtrackDetection('/home/jiahao/Downloads/data/Wildtrack'))
    wildtrack_data = Wildtrack(data_root, mask_type='people', mask_label_type='split', start_with=args.start_with, end_with=args.end_with)
    outputs = []
    for frame_idx in wildtrack_data.frame_idxs:
        print(f'Processing frame {frame_idx}')
        save_folder = os.path.join(data_root, 'depths', frame_idx)
        clustergs_path = os.path.join(save_folder, 'init_gs_3_1.ply') 
        # read the point cloud
        pcd = o3d.io.read_point_cloud(clustergs_path)
        gs_xyz = np.array(pcd.points)
        gs_rgb = np.array(pcd.colors)
        gs_xyz = wildtrack_detect_data.base.get_worldgrid_from_worldcoord(gs_xyz.T).T
        point_clouds = np.concatenate([gs_xyz, gs_rgb], axis=1)
        mask = (point_clouds[:, 0] > grid_range[0]) & (point_clouds[:, 0] < grid_range[3]) & \
                (point_clouds[:, 1] > grid_range[1]) & (point_clouds[:, 1] < grid_range[4]) #& \
                # (point_clouds[:, 2] > grid_range[2]) & (point_clouds[:, 2] < grid_range[5])
        point_clouds_filter = point_clouds[mask]
        
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(point_clouds_filter[:, :3])
        pcd.colors = o3d.utility.Vector3dVector(point_clouds_filter[:, 3:])
        o3d.visualization.draw_geometries([pcd])
        
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(point_clouds[:, :3])
        pcd.colors = o3d.utility.Vector3dVector(point_clouds[:, 3:])
        o3d.visualization.draw_geometries([pcd])
        
        
        
        # point_clouds = torch.from_numpy(point_clouds).float().to(device='cuda')
        # _, batch_pred = model({'point_clouds': point_clouds})
        # outputs.append(batch_pred['heatmap'].detach().cpu().numpy())
    return outputs

class Trainer(object):
    def __init__(self, model, args, device, summary, loss_weight=[1., 1.], gs_data_path=None, dataset=None) -> None:
        self.model = model
        self.args = args
        self.device = device
        self.summary = summary
        self.loss_weight = loss_weight
        self.monitor = Monitor()
        self.gs_data_apth = gs_data_path
        self.dataset = dataset
    def train(self, dataloader, optimizer, epoch, args):
        self.model.train()
        epoch_loss = MetricDict()
        t_b = time.time()
        t_forward, t_backward = 0, 0
        # wildtrack
        grid_range = [0, 0, 0, 480, 1440, 80]
        
        with tqdm(total=len(dataloader), desc=f'\033[33m[TRAIN]\033[0m Epoch {epoch} / {args.epochs}', postfix=dict, mininterval=0.2) as pbar:
            for idx, data in enumerate(dataloader):
        
                clustergs_path = os.path.join(self.gs_data_apth, 'depths', f"{data['frame_idx'].item():08d}", 'diffuse_gs_1_2.ply')
                pcd = o3d.io.read_point_cloud(clustergs_path)
                gs_xyz = np.array(pcd.points)
                gs_rgb = np.array(pcd.colors)
                gs_xyz = self.dataset.base.get_worldgrid_from_worldcoord_Tensor(gs_xyz)
                point_clouds = np.concatenate([gs_xyz, gs_rgb], axis=1)
        
                point_clouds = torch.from_numpy(point_clouds).float().to(device='cuda')
                data['point_clouds'] = point_clouds 
                
                batch_dict, batch_pred = self.model(data)
                
                loss, loss_dict = compute_loss(batch_pred, data, self.loss_weight)
                
                epoch_loss += loss_dict
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if idx % args.print_iter == 0:
                    mean_loss = epoch_loss.mean
                    pbar.set_postfix(**{
                        '(1)loss_total' : '\033[33m{:.6f}\033[0m'.format(mean_loss['loss']),
                        '(2)loss_heatmap' : '{:.5}'.format(mean_loss['loss_heatmap']),
                        }
                    )
                    pbar.update(1)
                if idx % args.vis_iter == 0:
                    steps = (epoch-1) * (len(dataloader) // args.vis_iter) + idx // args.vis_iter
                    heatmap_fig = visualize_heatmap(pred=torch.sigmoid(batch_pred['heatmap']).squeeze(dim=0).squeeze(dim=-1), gt=data['heatmap'].squeeze())
                    self.summary.add_figure('train/heatmap', heatmap_fig, steps)
                    monitor_fig = self.monitor.visualize(batch_dict, batch_pred, data, show=False)
                    monitor_fig[0].savefig("projection_res.jpg")
                    self.summary.add_figure('train/monitor', monitor_fig[0], steps)
                    plt.close('all')
            
        return epoch_loss.mean
    

    def validate(self, dataloader, epoch, args):
        self.model.eval()
        epoch_loss = MetricDict()
        t_b = time.time()
        t_forward, t_backward = 0, 0
        with tqdm(total=len(dataloader), desc=f'\033[31m[VAL]\033[0m Epoch {epoch} / {args.epochs}', postfix=dict, mininterval=0.2) as pbar:
            for idx, data in enumerate(dataloader):
                with torch.no_grad():
                    clustergs_path = os.path.join(self.gs_data_apth, 'depths', f"{data['frame_idx'].item():08d}", 'diffuse_gs_1_2.ply')
                    pcd = o3d.io.read_point_cloud(clustergs_path)
                    gs_xyz = np.array(pcd.points)
                    gs_rgb = np.array(pcd.colors)
                    gs_xyz = self.dataset.base.get_worldgrid_from_worldcoord_Tensor(gs_xyz.T).T
                    point_clouds = np.concatenate([gs_xyz, gs_rgb], axis=1)
                    point_clouds = torch.from_numpy(point_clouds).float().to(device='cuda')
                    data['point_clouds'] = point_clouds 
                    batch_dict, batch_pred = self.model(data)
                    
                    t_f = time.time()
                    t_forward += t_f - t_b
                    
                    loss, loss_dict = compute_loss(batch_pred, data, self.loss_weight)
                    
                    epoch_loss += loss_dict

                    t_b = time.time()
                    t_backward += t_b - t_f

                    if idx % args.print_iter == 0:
                        mean_loss = epoch_loss.mean
                        pbar.set_postfix(**{
                            '(1)loss_total' : '\033[33m{:.6f}\033[0m'.format(mean_loss['loss']),
                            '(2)loss_heatmap' : '{:.5}'.format(mean_loss['loss_heatmap']),
                            '(4)t_f & t_b' : '{:.2f} & {:.2f}'.format(t_forward/(idx+1), t_backward/(idx+1))
                            }
                        )
                        pbar.update(1)
        return epoch_loss.mean

def make_lr_scheduler(optimizer):
    w_iters = 5
    w_fac = 0.1
    max_iter = 40
    lr_lambda = lambda iteration : w_fac + (1 - w_fac) * iteration / w_iters \
            if iteration < w_iters \
            else 1 - (iteration - w_iters) / (max_iter - w_iters)
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda, last_epoch=-1)
    
    return scheduler

def setup_seed(seed=7777):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if use multi-GPU
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True
    np.random.seed(seed)
    random.seed(seed)

def make_experiment(args, cfg, copy_repo=False):
    lastdir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if cfg.DATA_CONFIG.DATASET == 'Wildtrack':
        lastdir += '_wt'
    elif cfg.DATA_CONFIG.DATASET == 'MultiviewX':
        lastdir += '_mx'
    args.savedir = os.path.join(args.savedir , lastdir)
    summary = SummaryWriter(args.savedir+'/tensorboard')
    summary.add_text('config', '\n'.join(
        '{:12s} {}'.format(k, v) for k, v in sorted(args.__dict__.items())))
    summary.file_writer.flush()
    if copy_repo:
        os.makedirs(args.savedir, exist_ok=True)
        copy_file(args.cfg_file, args.savedir)
    return summary, args

def resume_experiment(args):
    summary_dir = os.path.join(args.savedir, args.resume, 'tensorboard')
    args.savedir = os.path.join(args.savedir, args.resume)
    summary = SummaryWriter(summary_dir)
    return summary, args

def save(model, epoch, args, optimizer, scheduler, train_loss, val_loss):
    savedir = os.path.join(args.savedir, 'checkpoints')
    if not os.path.exists(savedir):
        os.mkdir(savedir)
    checkpoints = {
        'epoch' : epoch,
        'model_state_dict' : model.state_dict(),
        'optimizer_state_dict' : optimizer.state_dict(),
        'scheduler_state_dict' : scheduler.state_dict(),
        'args':args
    }
    torch.save(checkpoints, os.path.join(savedir, 'Epoch{:02d}_train_loss{:.4f}_val_loss{:.4f}.pth'.\
                        format(epoch, train_loss['loss'], val_loss['loss'])))

def resume(resume_dir, model, optimizer, scheduler, load_model_ckpt_only=False):
    checkpoints = torch.load(resume_dir)
    pretrain = checkpoints['model_state_dict']
    current = model.state_dict()
    state_dict = {k: v for k, v in pretrain.items() if k in current.keys()}
    current.update(state_dict)
    model.load_state_dict(current)
    if load_model_ckpt_only:
        return model, None, None, 1
    optimizer.load_state_dict(checkpoints['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoints['scheduler_state_dict'])
    epoch = checkpoints['epoch'] + 1
    print("Model resume training from %s" %resume_dir)
    return model, optimizer, scheduler, epoch

def parse_config(args_src=None):
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--cfg_file', type=str, default=r'submodules/MvCHM/cfgs/MvDDE.yaml',\
         help='specify the config for training')
    parser.add_argument('--workers', type=int, default=1, help='number of workers for dataloader')         
    
     # Training options
    parser.add_argument('-e', '--epochs', type=int, default=40,
                        help='the number of epochs for training')

    parser.add_argument('-b', '--batch_size', type=int, default=1,
                        help='batch size for training. [NOTICE]: this repo only support \
                              batch size of 1')

    parser.add_argument('--lr', type=float, default=0.0002,#0.0002,
                        help='learning rate')
    
    parser.add_argument('--weight_decay', type=float, default=5e-3,
                        help='learning rate')   

    parser.add_argument('--lr_step', type=list, default=[90, 120],
                        help='learning step')

    parser.add_argument('--lr_factor', type=float, default=0.1,
                        help='learning factor')
    
    parser.add_argument('--momentum', type=float, default=0.5,
                        help='SGD momentum')                        

    parser.add_argument('--savedir', type=str,
                        default='experiments')   

    parser.add_argument('--resume', type=str,
                        default=None)
    
    parser.add_argument('--checkpoint', type=str,
                        default=None)
    
    parser.add_argument('--sup_decoder_checkpoint', type=str,
                        default=None, help='The checkpoint of the supervised decoder - PointPillar')

    parser.add_argument('--print_iter', type=int, default=1,
                        help='print loss summary every N iterations')

    parser.add_argument('--vis_iter', type=int, default=100,
                        help='display visualizations every N iterations')      

    parser.add_argument('--loss_weight', type=float, default=[1., 1.],
                        help= '2D weight of each loss only including heatmap and location.')        

    parser.add_argument('--copy_yaml', type=bool, default=True,
                        help='Copy the whole repo before training')
    
    parser.add_argument('--pr_dir_pred', type=str, 
                        default=r'output/exp/pr_dir_pred.txt')
    parser.add_argument('--pr_dir_gt', type=str, 
                        default=r'output/exp/pr_dir_gt.txt')

    if args_src is not None:
        args = parser.parse_args(args_src)
    else:
        args = parser.parse_args()
        

    cfg_from_yaml_file(args.cfg_file, mvchm_cfg)

    return args, mvchm_cfg

def Train(args_src=None):
    DetectDataPath = '/home/jiahao/Downloads/data/Wildtrack'
    GSDataPath = '/home/jiahao/Downloads/data/wildtrack_data_gt'
    args, cfg = parse_config(args_src)

    # define devices
    device = torch.device('cuda:0')
    
    dataset_val = MvCHMMultiviewDataset(MvCHMWildtrack(DetectDataPath), set_name='val')
    val_dataloader = DataLoader( dataset_val, num_workers=1, batch_size=1)
    
    dataset_train = MvCHMMultiviewDataset(MvCHMWildtrack(DetectDataPath), set_name='train')
    train_dataloader = DataLoader( dataset_train, num_workers=1, batch_size=1)
    
    # define model
    model = PointPillar(cfg, torch.device('cuda'))

    optimizer = optim.Adam( model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.MultiStepLR( optimizer, args.lr_step, args.lr_factor )

    # Create Summary & Resume Training
    if args.resume is not None:
        summary, args = resume_experiment(args)
        resume_dir = os.path.join(args.savedir, 'checkpoints', args.checkpoint)
        # resume_dir = args.checkpoint
        model, optimizer, scheduler, start = \
            resume(resume_dir, model, optimizer, scheduler)
        args.epochs = args.epochs + 5
    else:
        summary, args = make_experiment(args, cfg, args.copy_yaml)
        start = 1
        
    trainer = Trainer(model, args, device, summary, args.loss_weight, GSDataPath, dataset_train)

    for epoch in range(start, args.epochs+1):
        summary.add_scalar('lr', optimizer.param_groups[0]['lr'], epoch)

        # Train model
        train_loss = trainer.train(train_dataloader, optimizer, epoch, args) 

        # Evaluate model
        val_loss = trainer.validate(val_dataloader, epoch, args)

        summary.add_scalars('loss', {'train_loss': train_loss['loss'], 'val_loss' : val_loss['loss']}, epoch)

        scheduler.step()

        if epoch % 1 == 0:
            save(model, epoch, args, optimizer, scheduler, train_loss, val_loss)

def heatmap_nms(prob):
    maxpool = nn.MaxPool2d(kernel_size=7, stride=1, padding=3)
    mask = torch.eq(prob, maxpool(prob)).to(prob.dtype)
    return prob * mask
    
            
def encode_postion(heatmap, mode, grid_reduce, thresh=None, nms=False, _mask=True, edge=20):
    if _mask:
        mask = torch.Tensor(np.load('n_mask.npy')).to(device=heatmap.device)
        mask = torch.where(mask < 1)
    assert mode in ['gt', 'pred']
    if len(heatmap.shape) != 2:
        heatmap=heatmap.squeeze(0).squeeze(-1)
    # heatmap_masks = torch.zeros_like(heatmap)
    # heatmap_masks[edge:-edge, edge:-edge] = 1
    # heatmap = heatmap * heatmap_masks
    # if len(offset_xy.shape) != 3:
    #     offset_xy = offset_xy.squeeze(0)
    if nms:
        heatmap = heatmap_nms(heatmap.unsqueeze(0).unsqueeze(0)).squeeze(0).squeeze(0)
    if mode == 'pred':
        heatmap = torch.sigmoid(heatmap)
        if _mask:
            heatmap[mask] = 0
    heatmap = to_numpy(heatmap)
    # offset_xy = to_numpy(offset_xy)
    if mode == 'gt':
        xx, yy = np.where(heatmap == 1)
    elif mode == 'pred':
        assert thresh is not None 
        xx, yy = np.where(heatmap >= thresh)
    # offset_xy = offset_xy[xx, yy]
    # pos_x = (xx + offset_xy[:,0]) * grid_reduce
    # pos_y = (yy + offset_xy[:,1]) * grid_reduce
    pos_x = xx * grid_reduce
    pos_y = yy * grid_reduce
    pos = np.stack([pos_x, pos_y, np.zeros_like(pos_x)], axis=-1)
    return pos

def main(args):
    if args.fun_type == 'Inference':
        Inference(args)
    elif args.fun_type == 'Train':
        Train(args)

if __name__ == '__main__':
    '''
    # Inference (For debugging)
    '''
    parser = ArgumentParser("Per scene training using Gaussian Splatting", add_help=True)
    parser.add_argument("--root", type=str, default='path_to_wildtrack_data_gt', help="path to image folder")
    parser.add_argument("--round", type=str, default='2_1', help="round name")
    parser.add_argument("--start-with", type=int, default=0, help="the index of the first image to start with")
    parser.add_argument("--end-with", type=int, default=-1, help="the index of the last image to end with")
    parser.add_argument("--n-segments", type=int, default=30, help="the number of superpixels for each person")
    parser.add_argument("--fun_type", choice=['Inference', 'Train', 'Evaluate'], default='Inference', help="function type to run")

    # FOR DEBUG
    args_src = [
                '--root', '/home/jiahao/Downloads/data/wildtrack_data_gt',
                '--round', '2_1',
                '--n-segments', '30',
                '--start-with', '0',
                '--end-with', '-1',
            ]
    # args = parser.parse_args(args_src) # FOR DEBUG
    args = parser.parse_args()
    
    # '''
    # # Inference
    # '''
    # Inference(args)
    
    # '''
    # # Training
    # '''
    # Train()

    # Wrap the main function to allow for easy execution
    main(args)
    
    
   
   