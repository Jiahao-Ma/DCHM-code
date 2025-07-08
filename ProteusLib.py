import os
import torch
import math
import numpy as np
import open3d as o3d
from torch import nn
from simple_knn._C import distCUDA2
from utils.sh_utils import RGB2SH
from utils.general_utils import inverse_sigmoid, mkdir_p, get_expon_lr_func
from plyfile import PlyData, PlyElement
from gsplat.rendering import rasterization, _rasterization
import matplotlib.pyplot as plt
from torchmetrics.functional.regression import pearson_corrcoef
from typing import Dict, List, Optional, Tuple, Union
from torch import Tensor
from sklearn.neighbors import NearestNeighbors
from submodules.gsplat.strategy import DefaultStrategy 
from submodules.customized_gsplat.rendering import rasterization_gs_id
class SuperPixelGaussian(nn.Module):
    def __init__(self, super_pixel_ply=None, max_sh_degree=3, required_grad=True, init_opacity=0.1):
        super(SuperPixelGaussian, self).__init__()
        self.max_sh_degree = max_sh_degree  # 1 or 3
        if super_pixel_ply is not None:
            self.load_from_ply(super_pixel_ply, required_grad, init_opacity=init_opacity)

        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log
        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid
        self.rotation_activation = torch.nn.functional.normalize

    def load_from_ckpt(self, path):
        plydata = PlyData.read(path)

        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])),  axis=1)
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

        features_dc = np.zeros((xyz.shape[0], 3, 1))
        features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

        extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
        extra_f_names = sorted(extra_f_names, key = lambda x: int(x.split('_')[-1]))
        assert len(extra_f_names)==3*(self.max_sh_degree + 1) ** 2 - 3
        features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
        for idx, attr_name in enumerate(extra_f_names):
            features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
        # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
        features_extra = features_extra.reshape((features_extra.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1))

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key = lambda x: int(x.split('_')[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rot_names = sorted(rot_names, key = lambda x: int(x.split('_')[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

        self._xyz = nn.Parameter(torch.tensor(xyz, dtype=torch.float, device="cuda").requires_grad_(True))
        self._features_dc = nn.Parameter(torch.tensor(features_dc, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(torch.tensor(features_extra, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(True))
        self._scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True))
        self._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True))

        self.active_sh_degree = self.max_sh_degree
    
    def load_from_ply(self, ply_path, required_grad=True, init_opacity=0.1):
        pcd = o3d.io.read_point_cloud(ply_path)
        fused_point_cloud = torch.tensor(np.asarray(pcd.points)).float().cuda()
        if len(pcd.colors) > 0:
            fused_color = RGB2SH(torch.tensor(np.asarray(pcd.colors)).float().cuda())
        else:
            fused_color = torch.zeros((fused_point_cloud.shape[0], 3)).float().cuda() # random color
        features = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()
        features[:, :3, 0 ] = fused_color
        features[:, 3:, 1:] = 0.0
        
        print("Number of points at initialisation : ", fused_point_cloud.shape[0])
        
        dist2 = torch.clamp_min(distCUDA2(torch.from_numpy(np.asarray(pcd.points)).float().cuda()), 0.0000001)
        scales = torch.log(torch.sqrt(dist2))[...,None].repeat(1, 3)
        rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
        rots[:, 0] = 1

        opacities = inverse_sigmoid(init_opacity * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))
        
        self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(required_grad))
        self._features_dc = nn.Parameter(features[:,:,0:1].transpose(1, 2).contiguous().requires_grad_(required_grad))
        self._features_rest = nn.Parameter(features[:,:,1:].transpose(1, 2).contiguous().requires_grad_(required_grad))
        self._scaling = nn.Parameter(scales.requires_grad_(required_grad))
        self._rotation = nn.Parameter(rots.requires_grad_(required_grad))
        self._opacity = nn.Parameter(opacities.requires_grad_(required_grad))
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")
        
    
    @property
    def get_bg(self):
        return self._bg
    
    @property
    def get_scaling(self):
        return self.scaling_activation(self._scaling)
    
    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation)
    
    @property
    def get_xyz(self):
        return self._xyz
    
    @property
    def get_features(self):
        features_dc = self._features_dc
        features_rest = self._features_rest
        return torch.cat((features_dc, features_rest), dim=1)
    
    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)
    
    def forward(self, x):
        pass
    
    def construct_list_of_attributes(self):
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        # All channels except the 3 DC
        for i in range(self._features_dc.shape[1]*self._features_dc.shape[2]):
            l.append('f_dc_{}'.format(i))
        for i in range(self._features_rest.shape[1]*self._features_rest.shape[2]):
            l.append('f_rest_{}'.format(i))
        l.append('opacity')
        for i in range(self._scaling.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(self._rotation.shape[1]):
            l.append('rot_{}'.format(i))
        return l
    
    def save_ply(self, path, save_type='all'):
        mkdir_p(os.path.dirname(path))
        if save_type=='added': 
            raise NotImplementedError
            xyz = self._added_xyz.detach().cpu().numpy()
            normals = np.zeros_like(xyz)
            f_dc = self._added_features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
            f_rest = self._added_features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
            opacities = self._added_opacity.detach().cpu().numpy()
            scale = self._added_scaling.detach().cpu().numpy()
            rotation = self._added_rotation.detach().cpu().numpy()       
        elif save_type=='origin':  
            xyz = self._xyz.detach().cpu().numpy()
            normals = np.zeros_like(xyz)
            f_dc = self._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
            f_rest = self._features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
            opacities = self._opacity.detach().cpu().numpy()
            scale = self._scaling.detach().cpu().numpy()
            rotation = self._rotation.detach().cpu().numpy()
        elif save_type=='all':
            xyz = self.get_xyz.detach().cpu().numpy()
            normals = np.zeros_like(xyz)
            f_dc = self.get_features[:,0:1,:].detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
            f_rest = self.get_features[:,1:,:].detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
            opacities = self.inverse_opacity_activation(self.get_opacity).detach().cpu().numpy()
            scale = self.scaling_inverse_activation(self.get_scaling).detach().cpu().numpy()
            rotation = self.get_rotation.detach().cpu().numpy()
        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]  
        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)
        
    def prune(self, min_opacity, extent, max_screen_size):
        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0
        prune_mask = (self.get_opacity < min_opacity).squeeze()
        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size
            big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * extent
            prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)
        self.prune_points(prune_mask)

        torch.cuda.empty_cache()
        
    def prune_points(self, mask):
        valid_points_mask = ~mask
        optimizable_tensors = self._prune_optimizer(valid_points_mask)

        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]

        self.denom = self.denom[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]
    
    def _prune_optimizer(self, mask):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter((group["params"][0][mask].requires_grad_(True)))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(group["params"][0][mask].requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors
   
    def training_setup(self, training_args):
        self.percent_dense = 0#training_args.percent_dense
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")

        l = [
            # {'params': [self._xyz], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "xyz"},
            {'params': [self._xyz], 'lr': training_args.position_lr_init , "name": "xyz"},
            {'params': [self._features_dc], 'lr': training_args.feature_lr, "name": "f_dc"},
            {'params': [self._features_rest], 'lr': training_args.feature_lr / 20.0, "name": "f_rest"},
            {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
            {'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
            {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"}
        ]

        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        self.xyz_scheduler_args = get_expon_lr_func(lr_init=training_args.position_lr_init,#*self.spatial_lr_scale,
                                                    lr_final=training_args.position_lr_final,#*self.spatial_lr_scale,
                                                    lr_delay_mult=training_args.position_lr_delay_mult,
                                                    max_steps=training_args.position_lr_max_steps)

    def update_learning_rate(self, iteration):
        ''' Learning rate scheduling per step '''
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "xyz":
                lr = self.xyz_scheduler_args(iteration)
                param_group['lr'] = lr
                return lr
            
            
class Proteus(nn.Module):
    def __init__(self, xyzs, colors, scales, max_sh_degree=3):
        super(Proteus, self).__init__()
        self.max_sh_degree = max_sh_degree
        self._init_component(xyzs, colors, scales)
    
    
    def _init_component(self, xyzs, colors, scales):
        fused_color = RGB2SH(colors)
        features = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()
        features[:, :3, 0 ] = fused_color
        features[:, 3:, 1:] = 0.0
        rots = torch.zeros((xyzs.shape[0], 4), device="cuda")
        rots[:, 0] = 1
        opacities = inverse_sigmoid(0.1 * torch.ones((xyzs.shape[0], 1), dtype=torch.float, device="cuda"))
        
        self._xyz = nn.Parameter(xyzs.requires_grad_(True))
        self._features_dc = nn.Parameter(features[:,:,0:1].transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(features[:,:,1:].transpose(1, 2).contiguous().requires_grad_(True))
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(True))
        self._opacity = nn.Parameter(opacities.requires_grad_(True))
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")
        
    @property
    def get_scaling(self):
        return self.scaling_activation(self._scaling)
    
    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation)
    
    @property
    def get_xyz(self):
        return self._xyz
    
    @property
    def get_features(self):
        features_dc = self._features_dc
        features_rest = self._features_rest
        return torch.cat((features_dc, features_rest), dim=1)
    
    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)
        
        
def render(pc: SuperPixelGaussian, cameras):
    means = pc.get_xyz.contiguous().view(-1, 3)
    colors = pc.get_features.contiguous()
    quats = pc.get_rotation.contiguous().view(-1, 4)
    scales = pc.get_scaling.contiguous().view(-1, 3)
    opacities = pc.get_opacity.contiguous().view(-1,)
    # convert the opacity from 0-1 to color space
    w2cs = torch.cat([cam['w2c'] for cam in cameras], dim=0).float().cuda()
    Ks = torch.cat([cam['K'] for cam in cameras], dim=0).float().cuda()
    width = cameras[0]['width'].item()
    height = cameras[0]['height'].item()

    render_colors, render_alphas, info = rasterization(
        means=means,
        quats=quats,
        scales=scales,
        opacities=opacities,
        colors=colors,
        viewmats=w2cs,  # [C, 4, 4]
        Ks=Ks,  # [C, 3, 3]
        width=width,
        height=height,
        packed=False,
        absgrad=False,
        sparse_grad=False,
        sh_degree=3,
        rasterize_mode='classic',
        render_mode='RGB+D'
    )
        
    return render_colors[..., :3], render_colors[..., 3:], render_alphas, info

def depth_constraints(depths, masks_split, cameras, cm2m=1e-2, loss_type='variance', random_select=5):
    assert loss_type in ['cluster', 'variance']
    losses = 0
    if loss_type == 'variance':
        for cam_idx in range(len(cameras)): 
            depth = depths[cam_idx].squeeze(-1)
            masks = masks_split[cam_idx]
            # randomly select some masks except the background `0`
            if random_select:
                mask_indices = torch.randperm(int(masks.max().item()))
                mask_indices = mask_indices[mask_indices != 0][:random_select]
            else:
                mask_indices = range(1, int(masks.max() + 1))
            for m_idx in mask_indices:
                depth_masked = depth[masks == m_idx]
                if depth_masked.size(0) > 1:
                    var = torch.var(depth_masked)
                else:
                    var = torch.tensor(0.0).to(depth.device)
                if torch.isnan(var):
                    continue
                if cm2m is not None:
                    var = var * cm2m
                losses += var
        return losses
    else:
        for cam_idx in range(len(cameras)):
            camera = cameras[cam_idx]
            height, width = camera['height'].item(), camera['width'].item()
            K = camera['K'].squeeze(0).cuda()
            c2w = camera['w2c'].squeeze(0).cuda()
            depth = depths[cam_idx].flatten()
            masks = masks_split[cam_idx]
            v, u = torch.meshgrid(torch.arange(height).cuda(), torch.arange(width).cuda())
            x = (u.flatten() - K[0, 2]) / K[0, 0] * depth
            y = (v.flatten() - K[1, 2]) / K[1, 1] * depth
            z = depth
            pts_cam_xyz = torch.stack([x, y, z], dim=-1)
            ptw_world_xyz = c2w[:3, :3] @ pts_cam_xyz.T + c2w[:3, 3:4]
            pts_world_xyz = ptw_world_xyz.T.contiguous().view(height, width, 3)
            if random_select:
                mask_indices = torch.randperm(int(masks.max().item()))
                mask_indices = mask_indices[mask_indices != 0][:random_select]
            else:
                mask_indices = range(1, int(masks.max() + 1))
            for m_idx in mask_indices:
                selected_xyz = pts_world_xyz[masks == m_idx, :]
                if selected_xyz.size(0) > 1:
                    var_x = torch.var(selected_xyz[:, 0]) 
                    var_y = torch.var(selected_xyz[:, 1])  
                else:
                    var_x = torch.tensor(0.0) 
                    var_y = torch.tensor(0.0)
                if torch.isnan(var_x) or torch.isnan(var_y):
                    continue
                if cm2m is not None:
                    var_x = var_x * cm2m
                    var_y = var_y * cm2m
                losses += var_x + var_y
        return losses
    

class DisparityToDepth(object):
    def __init__(self, eps=1e-4) -> None:
        self.__eps = eps
        
    def __call__(self, dis, rgb=None, depth_min=0.01, depth_max=20):
        depth = torch.zeros_like(dis, dtype=torch.float32)
        depth[dis>=self.__eps] = 1 / dis[dis>=self.__eps]
        mask = torch.logical_and(depth>=depth_min, depth<=depth_max)
        
        if rgb is not None:
            rgb_ = torch.zeros_like(rgb, dtype=np.float32)
            rgb_[dis>=self.__eps] = rgb[dis>=self.__eps]
            
            return depth, rgb_, mask.to(torch.bool)
        return depth, mask.to(torch.bool)
    

def pearson_correlation_loss(metric_depth, disparity, mask=None):
    """
    Compute Pearson correlation coefficient-based loss with optional mask.
    
    Args:
    pred (torch.Tensor): Predicted depth map
    target (torch.Tensor): Target depth map
    mask (torch.Tensor, optional): Binary mask (1 for valid pixels, 0 for invalid)
    
    Returns:
    torch.Tensor: Pearson correlation loss
    """
    def __forward(disparity, metric_depth):
        depth_loss = min(
                        (1 - pearson_corrcoef( - disparity, metric_depth)),
                        (1 - pearson_corrcoef(1 / (disparity + 200.), metric_depth))
        )
        return depth_loss
    if mask is not None:
        total_loss = 0
        for i in range(mask.shape[0]):
            m_d = metric_depth[i][mask[i]]
            dsp = disparity[i][mask[i]]
            dsp = dsp[m_d!=0]
            m_d = m_d[m_d!=0]
            total_loss += __forward(dsp, m_d)
    else:
        total_loss = __forward(disparity, metric_depth)
    return total_loss


def rgb_to_sh(rgb: Tensor) -> Tensor:
    C0 = 0.28209479177387814
    return (rgb - 0.5) / C0

def knn(x: Tensor, K: int = 4) -> Tensor:
    x_np = x.cpu().numpy()
    # find the nan and remove them
    nan_mask = np.isnan(x_np).any(axis=1)
    x_np = x_np[~nan_mask]
    model = NearestNeighbors(n_neighbors=K, metric="euclidean").fit(x_np)
    distances, _ = model.kneighbors(x_np)
    return torch.from_numpy(distances).to(x)

def init_splats_from_ply(ply_path, scene_scale, with_cluster_feat=False, cluster_feat_dim=16):
    pcd = o3d.io.read_point_cloud(ply_path)
    fused_point_cloud = np.asarray(pcd.points)
    fused_color = np.asarray(pcd.colors)
    if fused_color.shape[0] == 0:
        fused_color = np.random.rand(fused_point_cloud.shape[0], 3)
    return create_splats_with_optimizers(fused_point_cloud, fused_color, scene_scale=scene_scale, with_cluster_feat=with_cluster_feat, cluster_feat_dim=cluster_feat_dim)
    
def init_splats_from_pth(pth_path, device='cuda'):
    return torch.load(pth_path, map_location=device)["splats"]

def create_splats_with_optimizers(
    points,
    points_rgb,
    init_type: str = "sfm",
    init_num_pts: int = 100_000,
    init_extent: float = 3.0,
    init_opacity: float = 0.1,
    init_scale: float = 1.0,
    scene_scale: float = 1.0,
    sh_degree: int = 3,
    sparse_grad: bool = False,
    batch_size: int = 1,
    feature_dim: Optional[int] = None,
    device: str = "cuda",
    world_rank: int = 0,
    world_size: int = 1,
    with_cluster_feat: bool = False,
    cluster_feat_dim: int = 16,
) -> Tuple[torch.nn.ParameterDict, Dict[str, torch.optim.Optimizer]]:
    if init_type == "sfm":
        points = torch.from_numpy(points).float()
        rgbs = torch.from_numpy(points_rgb / 255.0).float()
    elif init_type == "random":
        points = init_extent * scene_scale * (torch.rand((init_num_pts, 3)) * 2 - 1)
        rgbs = torch.rand((init_num_pts, 3))
    else:
        raise ValueError("Please specify a correct init_type: sfm or random")
    nan_mask = torch.isnan(points).any(axis=1)
    points = points[~nan_mask]
    rgbs = rgbs[~nan_mask]
    
    # Initialize the GS size to be the average dist of the 3 nearest neighbors
    dist2_avg = (knn(points, 4)[:, 1:] ** 2).mean(dim=-1)  # [N,]
    dist_avg = torch.sqrt(dist2_avg)
    scales = torch.log(dist_avg * init_scale).unsqueeze(-1).repeat(1, 3)  # [N, 3]
    
    # Distribute the GSs to different ranks (also works for single rank)
    points = points[world_rank::world_size]
    rgbs = rgbs[world_rank::world_size]
    scales = scales[world_rank::world_size]

    N = points.shape[0]
    quats = torch.rand((N, 4))  # [N, 4]
    opacities = torch.logit(torch.full((N,), init_opacity))  # [N,]

    params = [
        # name, value, lr
        ("means", torch.nn.Parameter(points), 1.6e-4 * scene_scale),
        ("scales", torch.nn.Parameter(scales), 5e-3),
        ("quats", torch.nn.Parameter(quats), 1e-3),
        ("opacities", torch.nn.Parameter(opacities), 5e-2),
    ]

    if feature_dim is None:
        # color is SH coefficients.
        colors = torch.zeros((N, (sh_degree + 1) ** 2, 3))  # [N, K, 3]
        colors[:, 0, :] = rgb_to_sh(rgbs)
        params.append(("sh0", torch.nn.Parameter(colors[:, :1, :]), 2.5e-3))
        params.append(("shN", torch.nn.Parameter(colors[:, 1:, :]), 2.5e-3 / 20))
    else:
        # features will be used for appearance and view-dependent shading
        features = torch.rand(N, feature_dim)  # [N, feature_dim]
        params.append(("features", torch.nn.Parameter(features), 2.5e-3))
        colors = torch.logit(rgbs)  # [N, 3]
        params.append(("colors", torch.nn.Parameter(colors), 2.5e-3))
        
    if with_cluster_feat:
        cluster_feat = torch.rand(N, cluster_feat_dim)
        params.append(("cluster_feat", torch.nn.Parameter(cluster_feat), 2.5e-3))

    splats = torch.nn.ParameterDict({n: v for n, v, _ in params}).to(device)
    # Scale learning rate based on batch size, reference:
    # https://www.cs.princeton.edu/~smalladi/blog/2024/01/22/SDEs-ScalingRules/
    # Note that this would not make the training exactly equivalent, see
    # https://arxiv.org/pdf/2402.18824v1
    BS = batch_size * world_size
    optimizers = {
        name: (torch.optim.SparseAdam if sparse_grad else torch.optim.Adam)(
            [{"params": splats[name], "lr": lr * math.sqrt(BS), "name": name}],
            eps=1e-15 / math.sqrt(BS),
            # TODO: check betas logic when BS is larger than 10 betas[0] will be zero.
            betas=(1 - BS * (1 - 0.9), 1 - BS * (1 - 0.999)),
        )
        for name, _, lr in params
    }
    return splats, optimizers

def rasterize_splats(
        splats,
        w2c: Tensor,
        Ks: Tensor,
        width: int,
        height: int,
        render_mode="RGB+D",
        sh_degree:int=3, 
        rasterize_feat: bool = False
    ) -> Tuple[Tensor, Tensor, Dict]:
        means = splats["means"]  # [N, 3]
        # quats = F.normalize(self.splats["quats"], dim=-1)  # [N, 4]
        # rasterization does normalization internally
        quats = splats["quats"]  # [N, 4]
        scales = torch.exp(splats["scales"])  # [N, 3]
        opacities = torch.sigmoid(splats["opacities"])  # [N,]
        if rasterize_feat:
            colors = splats["cluster_feat"] # [N, K]
        else:
            colors = torch.cat([splats["sh0"], splats["shN"]], 1)  # [N, K, 3]
        # # pytorch version of rasterization
        # render_colors, render_alphas, info = _rasterization(
        #     means=means,
        #     quats=quats,
        #     scales=scales,
        #     opacities=opacities,
        #     colors=colors,
        #     viewmats=w2c,  # [C, 4, 4]
        #     Ks=Ks,  # [C, 3, 3]
        #     width=width,
        #     height=height,
        #     render_mode=render_mode,
        #     sh_degree=sh_degree,
        # )
        render_colors, render_alphas, info = rasterization(
            means=means,
            quats=quats,
            scales=scales,
            opacities=opacities,
            colors=colors,
            viewmats=w2c,  # [C, 4, 4]
            Ks=Ks,  # [C, 3, 3]
            width=width,
            height=height,
            render_mode=render_mode,
            sh_degree=sh_degree,
            packed=False
        )
        return render_colors, render_alphas, info

def rasterize_splats_id(
        splats,
        w2c: Tensor,
        Ks: Tensor,
        width: int,
        height: int,
    ) -> Tuple[Tensor, Tensor, Dict]:
        means = splats["means"]  # [N, 3]
        # quats = F.normalize(self.splats["quats"], dim=-1)  # [N, 4]
        # rasterization does normalization internally
        quats = splats["quats"]  # [N, 4]
        scales = torch.exp(splats["scales"])  # [N, 3]
        opacities = torch.sigmoid(splats["opacities"])  # [N,]

        # pytorch version of rasterization
        with torch.no_grad():
            weights, gs_ids, pixel_ids, camera_ids, meta = rasterization_gs_id(
                means=means,
                quats=quats,
                scales=scales,
                opacities=opacities,
                viewmats=w2c,  # [C, 4, 4]
                Ks=Ks,  # [C, 3, 3]
                width=width,
                height=height,
            )
        return weights, gs_ids, pixel_ids, camera_ids, meta
