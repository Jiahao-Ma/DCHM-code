import time
import sys, os
sys.path.append("/home/dzc/Projects/MvCHM")
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import torch
import torch.nn as nn
import numpy as np
import random
import cv2
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import Sampler
from torchvision import transforms
import torch.nn.functional as F
from lib.data.wildtrack import Wildtrack
from lib.data.multiviewX import MultiviewX
from model.ffe.ray import getBBoxDepthMap, c2w_cvt
class MultiviewDataset(Dataset):
    def __init__(self, base:Wildtrack, 
                       set_name:str, 
                       transform_img:object=None,
                       transform_annot:object=None,
                       transform_depth:object=None,
                       split_ratio:float=0.9,
                       scale=None, # NOTICE: if the image is resize, intrinsice mat need to change as well.
                       new_h = 720,
                       new_w = 1280,
                       pad_h = 0, pad_w = 0
                       ) -> None:
        super().__init__()
        assert set_name in ['train', 'val', 'all'], 'split mode error'
        assert isinstance(base, Wildtrack) or isinstance(base, MultiviewX), 'base dataset error'
        self.base = base
        self.__name__ = base.__name__
        self.img_size = base.img_size
        self.transform_img = transform_img
        self.transform_annot = transform_annot
        self.labels_bbox, self.heatmaps, self.offset_xy, self.fpaths, self.labels_pos = self.split(
            set_name, split_ratio, base.labels_bbox, base.heatmaps, base.offset_xy, base.labels_pos)
        self.intrinsic_matrices, self.extrinsic_matrices = base.intrinsic_matrices, base.extrinsic_matrices
        if scale is not None:
            self.ori_intrinsic_matrices = self.intrinsic_matrices
            self.intrinsic_matrices = [scale @ K for K in self.intrinsic_matrices]
        self.labels = {0: 'pedestrian'}
        self.depth_map, self.depth_max, self.grid_range = base.depth_map, base.depth_max, base.grid_range
        if isinstance(base, MultiviewX):
            self.depth_min = base.depth_min
        if transform_depth is not None:
            self.depth_map = torch.stack(self.depth_map, dim=0)
            self.depth_map = transform_depth( self.depth_map )
        self.new_h, self.new_w = new_h, new_w
        self.pad_h, self.pad_w = pad_h, pad_w

    def split(self, set_name, split_ratio, labels_bbox, heatmaps, offset_xy, labels_pos=None):
        """
            Split the labels(annotations), heatmap( the pedestrians' position on the ground ) and 
            the path of image base on set_name. Train set occupies 90%, val set occupies 10%.  
        """
        assert len(labels_bbox) == len(heatmaps) == len(offset_xy)
        if set_name == 'train':
            if self.base.__name__ == Wildtrack.__name__:
                self.frame_range = range(0, int(self.base.num_frame * split_ratio), 5)
            elif self.base.__name__ == MultiviewX.__name__:
                self.frame_range = range(0, int(self.base.num_frame * split_ratio))
        elif set_name == 'val':
            if self.base.__name__ == Wildtrack.__name__:
                self.frame_range = range(int(self.base.num_frame * split_ratio), int(self.base.num_frame), 5)
            elif self.base.__name__ == MultiviewX.__name__:
                self.frame_range = range(int(self.base.num_frame * split_ratio), int(self.base.num_frame))
        elif set_name == 'all':
            if self.base.__name__ == Wildtrack.__name__:
                self.frame_range = range(0, int(self.base.num_frame), 5)
            elif self.base.__name__ == MultiviewX.__name__:
                self.frame_range = range(0, int(self.base.num_frame))

        if self.base.__name__ == Wildtrack.__name__:
            labels_bbox = [labels_bbox[id] for id, i in enumerate(range(0, int(self.base.num_frame), 5)) if i in self.frame_range]
            heatmaps = [heatmaps[id] for id, i in enumerate(range(0, int(self.base.num_frame), 5)) if i in self.frame_range]
            offset_xy = [offset_xy[id] for id, i in enumerate(range(0, int(self.base.num_frame), 5)) if i in self.frame_range]
            if labels_pos is not None:
                labels_pos = [labels_pos[id] for id, i in enumerate(range(0, int(self.base.num_frame), 5)) if i in self.frame_range]
        elif self.base.__name__ == MultiviewX.__name__:
            labels_bbox = [labels_bbox[id] for id, i in enumerate(range(0, int(self.base.num_frame))) if i in self.frame_range]
            heatmaps = [heatmaps[id] for id, i in enumerate(range(0, int(self.base.num_frame))) if i in self.frame_range]
            offset_xy = [offset_xy[id] for id, i in enumerate(range(0, int(self.base.num_frame))) if i in self.frame_range]
            if labels_pos is not None:
                labels_pos = [labels_pos[id] for id, i in enumerate(range(0, int(self.base.num_frame))) if i in self.frame_range]
        
        fpaths = self.base.get_image_fpaths(self.frame_range)
        self.frame_range = list(self.frame_range)
        return labels_bbox, heatmaps, offset_xy, fpaths, labels_pos
    
    def __len__(self):
        return len(self.frame_range)
    
    def __getitem__(self, idx):
        imgs = self.load_image2(idx)
        annots = self.load_annotations(idx)
        if len(annots) == 5:
            annot, heatmap, pos, keypoints = annots
            sample = {'img':imgs, 'annot':annot, 'heatmap':heatmap, 'pos':pos, 'keypoints':keypoints}
        else:
            annot, heatmap, pos = annots
            sample = {'img':imgs, 'annot':annot, 'heatmap':heatmap, 'pos':pos, 'keypoints':None}
        
        if self.transform_img:
            """
            Image ops: Resize -> Pad -> Normalize
            Annotation ops: Resize 
            Heatmap ops: numpy -> tensor
            """
            sample['ori_img'] = [transforms.functional.pil_to_tensor(img) for img in sample['img'].copy()]
            sample['img'] = [self.transform_img(img) for img in sample['img']] # shape (3, 384, 640) # 2
        if self.transform_annot:
            sample['ori_annot'] = [np.array(annot) for annot in sample['annot'].copy()]
            sample['annot'] = [self.transform_annot(annot) for annot in sample['annot']]
            sample['heatmap'] = torch.from_numpy(sample['heatmap'])
            if sample['keypoints'] is not None:
                sample['keypoints'] = [ torch.Tensor(kyp) for kyp in sample['keypoints']]
        return sample

    def load_image(self, idx):
        img_path_list = [self.fpaths[cam][self.frame_range[idx]] for cam in range(1 , self.base.num_cam+1)]
        img_list = []
        for idx, path in enumerate(img_path_list):
            img = cv2.imread(path, cv2.IMREAD_COLOR)
            img = cv2.resize(img, (self.new_w, self.new_h), interpolation=cv2.INTER_LINEAR)
            img = img.transpose(2, 0, 1)
        return img

    def load_image2(self, idx):
        img_path_list = [self.fpaths[cam][self.frame_range[idx]] for cam in range(1 , self.base.num_cam+1)]
        img_list = [Image.open(path) for path in img_path_list]
        return img_list

    def load_annotations(self, idx):
        labels_bbox = [ annot for annot in self.labels_bbox[idx].values()]
        labels_pos = [annot for annot in self.labels_pos[idx].values()]
        heatmap = self.heatmaps[idx]
        return labels_bbox, heatmap, labels_pos
    
    def num_classes(self):
        return 1 # only pedestrian

def collater(data):
    """
        This collater is for MultiviewX and Wildtrack dataset.
        Args:
            data {  'img': a list that contains the image array
                    'annot': a list that contains the annotation
                    'scale': the scale of image
                    'heatmap': the gt of pedestrian's position on the ground
                }
        Returns:
            sample { 'img':  torch.Tensor with the shape of [num_cam, channels, height, width]
                     'annot': torch.Tensor contains the information of annotation, with the 
                              shape of [num_cam, max_num_annot, 5]. The empty tensor fills with -1.
                     'scale': remain the same
                     'heatmap': torch.Tensor contains the gt of pedestrian's position
                                on the ground.
            }
    """
    imgs_tensor = torch.stack(data[0]['img'], dim=0)
    ori_img = torch.stack(data[0]['ori_img'], dim=0)
    annot = data[0]['annot']
    max_num_annots = max([a.shape[0] for a in annot[0]])
    
    num_cam = len(annot[0])
    if max_num_annots > 0:
        labels_bbox_tensor = torch.ones((num_cam, max_num_annots, 6)) * -1
        for idx, bbox in enumerate(annot[0]):
            if bbox.shape[0] > 0:
                labels_bbox_tensor[idx, :bbox.shape[0], :] = bbox
    else:
        labels_bbox_tensor = torch.ones((num_cam, 1, 6)) * -1
    return {'img': imgs_tensor, 'ori_img': ori_img, 'annot': labels_bbox_tensor, 'ori_annot': data[0]['ori_annot'], 'heatmap': data[0]['heatmap']}

class TransformAnnot(nn.Module):
    def __init__(self, scale_h=None, scale_w=None) -> None:
        super().__init__()
        self.scale_h = scale_h
        self.scale_w = scale_w
    
    def forward(self, annots):
        annots[:, [0, 2, 4]] *= self.scale_w
        annots[:, [1, 3, 5]] *= self.scale_h
        return torch.from_numpy(annots)


class UnNormalizer(object):
    def __init__(self, mean=None, std=None):
        if mean is None:
            self.mean = [0.485, 0.456, 0.406]
        else:
            self.mean = mean
        if std is None:
            self.std = [0.229, 0.224, 0.225]
        else:
            self.std = std

    def __call__(self, tensor):
        """ 
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
        return tensor


class AspectRatioBasedSampler(Sampler):

    def __init__(self, data_source, batch_size, drop_last):
        self.data_source = data_source
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.groups = self.group_images()

    def __iter__(self):
        random.shuffle(self.groups)
        for group in self.groups:
            yield group

    def __len__(self):
        if self.drop_last:
            return len(self.data_source) // self.batch_size
        else:
            return (len(self.data_source) + self.batch_size - 1) // self.batch_size

    def group_images(self):
        # determine the order of the images
        order = list(range(len(self.data_source)))
        order.sort(key=lambda x: self.data_source.image_aspect_ratio(x))

        # divide into groups, one group = one batch
        return [[order[x % len(order)] for x in range(i, i + self.batch_size)] for i in range(0, len(order), self.batch_size)]

class ResizeV14(nn.Module):
    def __init__(self, size, interpolation='bilinear', align_corners=False) -> None:
        super().__init__()
        assert interpolation in ['nearest', 'nearest', 'linear', 
                     'bilinear', 'bicubic', 'trilinear', 'area']
        self.size=size
        self.interpolation = interpolation
        self.align_corners = align_corners
    def forward(self, x):
        # (n, h, w) -> (1, n, h, w) -> (1, n, h`, w`) -> (n, h`, w`)
        x = torch.unsqueeze(x, dim=0)
        x = F.interpolate(x, size=self.size, mode=self.interpolation, align_corners=False)
        return torch.squeeze(x, dim=0)

class PadV14(nn.Module):
    def __init__(self, padding, fill=0, padding_mode='constant'):
        super().__init__()
        self.padding = padding
        self.fill = fill 
        self.padding_mode = padding_mode

    def forward(self, x):
        # Version: 1.4  F.pad: (left, right, top, bottom)
        return F.pad(input=x, pad=self.padding, value=self.fill, mode=self.padding_mode)

def get_padded_value(h, w, multiple_number=64):
    pad_h = (h + multiple_number - 1) // multiple_number * multiple_number
    pad_w = (w + multiple_number - 1) // multiple_number * multiple_number
    pad_h -= h
    pad_w -= w
    return pad_h, pad_w

