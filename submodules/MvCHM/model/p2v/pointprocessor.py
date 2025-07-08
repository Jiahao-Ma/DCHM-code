import torch
import torch.nn as nn
import numpy as np
from lib.utils.tool_utils import to_numpy
from model.ffe.pcl import write_point_clouds
from model.p2v.point_cloud_ops import points_to_voxel
from time import time
from copy import deepcopy
tv = None

try:
    import cumm.tensorview as tv
except:
    pass

class VoxelGeneratorWrapper(object):
    def __init__(self, implement, vsize_xyz, coors_range_xyz, num_point_features, max_num_points_per_voxel, max_num_voxels, keep_point_cloud_rate=None, unnormalize=None):
        self.implement = implement
        if self.implement == 'spconv':
            self.voxel_params = {   'vsize_xyz'                : vsize_xyz,
                                    'coors_range_xyz'          : coors_range_xyz,
                                    'num_point_features'       : num_point_features,
                                    'max_num_points_per_voxel' : max_num_points_per_voxel,
                                    'max_num_voxels'           : max_num_voxels, 
                                    
                                }
            from spconv.utils import Point2VoxelCPU3d as VoxelGenerator  # NOTICE: Version of Spconv is 2.x
            self._voxel_generator = VoxelGenerator( **self.voxel_params )
        else:
            self.voxel_params = { 'voxel_size'   : vsize_xyz,
                                  'coors_range'  : coors_range_xyz,
                                  'max_points'   : max_num_points_per_voxel,
                                  'max_voxels'   : max_num_voxels,
                                }
        self.keep_point_cloud_rate = keep_point_cloud_rate
        self.unnormalize = unnormalize

    def generate(self, points):
        device = points.device
        if self.keep_point_cloud_rate is not False:
            points=self.random_remove_point_cloud(points, self.keep_point_cloud_rate)

        if self.implement == 'spconv':
            assert tv is not None, f"Unexpected error, library: 'cumm' wasn't imported properly."
            voxel_output = self._voxel_generator.point_to_voxel(tv.from_numpy(to_numpy(points)))
            voxels, coordinates, num_points = voxel_output
            # make copy with numpy(), since numpy_view() will disappear as soon as the generator is deleted
        else:
            voxel_output = points_to_voxel(points=to_numpy(points), **self.voxel_params)
            voxels, coordinates, num_points = voxel_output
            voxels = torch.Tensor(voxels).to(device=device)
            coordinates = torch.Tensor(coordinates).to(device=device)
            num_points = torch.Tensor(num_points).to(device=device)
        return voxels, coordinates, num_points
    
    def random_remove_point_cloud(self, points, rate):
        num_pc = points.shape[0]
        index_pc = torch.randperm(num_pc)
        points = points[index_pc][:int(num_pc * rate)]
        return points

class PointProcesspr(nn.Module):
    def __init__(self, config, unnormalize):
        super().__init__()
        self.model_cfg = config.MODEL
        self.dataname = config.DATA_CONFIG.DATASET
        mode = 'train' # this param is used to determine the maximum number of voxels, the number of `training` mode is the same as that of 'test' mode.
        self.unnormalize = unnormalize
        self.grid_range = config.DATA_CONFIG.GRID_RANGE
        self.data_processor_queue = list()
        for cur_cfg in self.model_cfg.P2V.DATA_PROCESSOR:
            if not cur_cfg.ENABLE:
                continue
            cur_processor = getattr(self, cur_cfg.NAME)
            self.data_processor_queue.append(cur_processor)
            if cur_cfg.NAME == 'transform_points_to_voxel':
                self.voxel_generator = VoxelGeneratorWrapper(implement=cur_cfg.IMPLEMENT,
                                                             vsize_xyz=config.DATA_CONFIG.VOXEL_SIZE,
                                                             coors_range_xyz=self.grid_range,
                                                             num_point_features=cur_cfg.NUM_POINT_FEATURES,
                                                             max_num_points_per_voxel=cur_cfg.MAX_POINTS_PER_VOXEL,
                                                             max_num_voxels=cur_cfg.MAX_NUMBER_OF_VOXELS[mode],
                                                             keep_point_cloud_rate=cur_cfg.KEEP_POINT_CLOUD_RATE,
                                                            )
    

            
    def mask_points_outside_range(self, batch_dict, save_pc=False):
        assert batch_dict['point_clouds'] is not None, 'point coulds are empty.'
        points = batch_dict['point_clouds']
        mask = (points[:, 0] >= self.grid_range[0]) & (points[:, 0] <= self.grid_range[3]) & \
               (points[:, 1] >= self.grid_range[1]) & (points[:, 1] <= self.grid_range[4]) & \
               (points[:, 2] >= self.grid_range[2]) & (points[:, 2] <= self.grid_range[5]) 
        batch_dict['point_clouds'] = points[mask]
        if save_pc: # for evaluation
            temp_points = points.clone()
            if self.unnormalize is not None:
                temp_points[:, 3:6] = self.unnormalize(temp_points[:, 3:6]) * 255
            else:
                temp_points[:, 3:6] = temp_points[:, 3:6] * 255
            temp_points[:, 3:6] = torch.clamp(temp_points[:, 3:6], min=0, max=255)
            write_point_clouds('./visualization/pc/points_wo_filtered.ply', temp_points.cpu().numpy())
            write_point_clouds('./visualization/pc/points_w_filtered.ply', temp_points[mask].cpu().numpy())
        return batch_dict
    
    def shuffle_points(self, batch_dict):
        assert batch_dict['point_clouds'] is not None, 'point coulds are empty.'
        points = batch_dict['point_clouds']
        shuffle_idx = np.random.permutation(points.shape[0])
        batch_dict['point_clouds'] = points[shuffle_idx]
        return batch_dict
    
    def transform_points_to_voxel(self, batch_dict):
        assert batch_dict['point_clouds'] is not None, 'point coulds are empty.'
        points = batch_dict['point_clouds']
        voxel_output = self.voxel_generator.generate(points)
        voxels, coordinates, num_points = voxel_output
        batch_dict['voxels'] = voxels
        batch_dict['voxel_coords'] = coordinates
        batch_dict['voxel_num_points'] = num_points
        return batch_dict

    def forward(self, batch_dict):
        if self.dataname == 'MultiviewX':
            # MultiviewX only xy -> yx
            points = batch_dict['point_clouds']
            px = points[:, 0].clone()
            py = points[:, 1].clone()
            points[:, 0] = py
            points[:, 1] = px
            batch_dict['point_clouds'] = points 
        for cur_processor in self.data_processor_queue:
            batch_dict = cur_processor(batch_dict)
        return batch_dict