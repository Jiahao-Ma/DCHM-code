import torch
import torch.nn as nn

class PointPillarScatter(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.model_cfg = config.MODEL
        self.num_bev_features = config.MODEL.MAP_TO_BEV.NUM_BEV_FEATURES
        grid_size = config.DATA_CONFIG.GRID_RANGE[-3:]
        voxel_size = config.DATA_CONFIG.VOXEL_SIZE
        self.nx = grid_size[0] // voxel_size[0] # 120
        self.ny = grid_size[1] // voxel_size[1] # 360 
        self.nz = grid_size[2] // voxel_size[2] # 1
        assert self.nz == 1

    def forward(self, batch_dict):
        pillar_features, coord = batch_dict['pillar_features'], batch_dict['voxel_coords'] # coords: z y x
        spatial_feature = torch.zeros(
                                      self.num_bev_features,
                                      self.nz * self.ny * self.nx,
                                      dtype=pillar_features.dtype,
                                      device=pillar_features.device) # used to contain pillar features, shape [64, 43200]

        indices = coord[:, 0] + coord[:, 1] + coord[:, 2] * self.ny
        indices = indices.type(torch.long)
        pillar_features = pillar_features.t()
        spatial_feature[:, indices] = pillar_features  # pillar_features [64, 1502]
        spatial_feature = spatial_feature.view(self.num_bev_features * self.nz, self.nx, self.ny) # 64, 120, 360
        batch_dict['spatial_features'] = torch.unsqueeze(spatial_feature, dim=0) # 1, 64, 120, 360
        return batch_dict


