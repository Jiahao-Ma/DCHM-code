import torch.nn as nn

from model.p2v.pointprocessor import PointProcesspr

class Point2Voxel(nn.Module):
    def __init__(self, model_cfg, unnormalize=None):
        super().__init__()
        self.point_processor = PointProcesspr(model_cfg, unnormalize)


    def forward(self, batch_dict):
        batch_dict = self.point_processor(batch_dict)
        return batch_dict

