import torch.nn as nn
import os, sys; sys.path.append(os.getcwd())

from .DCNv2 import DeformConv2d
class Head(nn.Module):
    def __init__(self, config):
        super().__init__()
        in_channels = config.MODEL.HEAD.IN_CHANNELS
        # self.conv_pos_origin = nn.Sequential(
        #     nn.Conv2d( in_channels=in_channels, out_channels=in_channels, kernel_size=3, padding=4, dilation=4, bias=False),
        #     nn.BatchNorm2d(in_channels), nn.ReLU(True),
        #     DeformConv2d(in_channels, in_channels, 2, padding=2),
        #     nn.BatchNorm2d(in_channels), nn.ReLU(True),
        #     nn.Conv2d( in_channels=in_channels, out_channels=1, kernel_size=3, padding=1, bias=False)
        #     )

        # self.conv_pos_93_1 = nn.Sequential(
        #     nn.Conv2d( in_channels=in_channels, out_channels=in_channels, kernel_size=3, padding=1, bias=False),
        #     nn.BatchNorm2d(in_channels), nn.ReLU(True),
        #     DeformConv2d(in_channels, in_channels, 3, padding=1),
        #     nn.BatchNorm2d(in_channels), nn.ReLU(True),
        #     nn.Conv2d(in_channels=in_channels, out_channels=1, kernel_size=3, padding=4, dilation=4, bias=False)
        #     )

        # MODA 93.4, 2022-08-17-11-37--epoch22
        # self.conv_pos = nn.Sequential(
        #     nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=3, padding=4, dilation=4, bias=False),
        #     nn.LeakyReLU(True),
        #     DeformConv2d(in_channels, in_channels, 2, padding=2),
        #     nn.LeakyReLU(True),
        #     nn.Conv2d(in_channels=in_channels, out_channels=1, kernel_size=3, padding=1, bias=False)
        #     )

        # MODA 94.1, 2022-08-18 --epoch10 thresh 0.9, lr 0.0002 normal_nms
        # MODA 94.3, 2022-08-18 --epoch10 thresh 0.9, lr 0.0002 normal_nms thresh 0.5 0.4
        # self.conv_pos = nn.Sequential(
        #     nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=3, padding=1, bias=False),nn.ReLU(),
        #     DeformConv2d(in_channels, in_channels, 2, padding=2),nn.ReLU(),
        #     nn.Conv2d(in_channels, in_channels, 3, padding=2, dilation=2), nn.ReLU(),
        #     nn.Conv2d(in_channels=in_channels, out_channels=1, kernel_size=3, padding=4, dilation=4, bias=False),
        #     )

        self.conv_pos = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=3, padding=1, bias=False),nn.ReLU(),
            # DeformConv2d(in_channels, in_channels, 2, padding=2),nn.LeakyReLU(),
            nn.Conv2d(in_channels, in_channels, 3, padding=2, dilation=2),nn.BatchNorm2d(in_channels), nn.ReLU(),
            nn.Conv2d(in_channels=in_channels, out_channels=1, kernel_size=3, padding=4, dilation=4, bias=False),
            )

        # self.conv_off = nn.Sequential( 
        #     nn.Conv2d( in_channels=in_channels, out_channels=in_channels, kernel_size=3, padding=4, dilation=4, bias=False),                 
        #     nn.BatchNorm2d(in_channels), nn.ReLU(True),
        #     nn.Conv2d( in_channels=in_channels, out_channels=2, kernel_size=3, padding=1, bias=False)
        #     )
       
    def forward(self, batch_dict):
        compact_spatial_features = batch_dict['compact_spatial_features']
        pos_preds = self.conv_pos(compact_spatial_features)
        # off_preds = self.conv_off(compact_spatial_features)

        pos_preds = pos_preds.permute(0, 2, 3, 1).contiguous() # (bz, l, w, 1)
        # off_preds = off_preds.permute(0, 2, 3, 1).contiguous() # (bz, l, w, 2)
        
        return {'heatmap': pos_preds, 'offset': None}
