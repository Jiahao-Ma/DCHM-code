import os
import torch
import numpy as np
from collections import namedtuple
from collections import defaultdict
import torch.nn.functional as F
def to_numpy(data):
    if isinstance(data, np.ndarray):
        return data
    elif isinstance(data, torch.Tensor):
        return data.detach().cpu().numpy()
    else:
        return np.array(data)

def to_tensor(data):
    if isinstance(data, torch.Tensor):
        return data
    elif isinstance(data, np.ndarray):
        return torch.Tensor(data)
   
def make_grid(world_size=(3900, 3900), grid_offset=(0, 0, 0), cube_LW=[25, 25], dataset='Wildtrack'):
    if dataset == 'Wildtrack':
        width, length = world_size
    elif dataset == 'MultiviewX':
        length, width = world_size
        
        
    xoff, yoff, zoff = grid_offset
    xcoords = torch.arange(0., width, cube_LW[0]) + xoff
    ycoords = torch.arange(0., length, cube_LW[1]) + yoff

    xx, yy = torch.meshgrid([xcoords, ycoords])
    
    return torch.stack([xx, yy, torch.full_like(xx, zoff)], dim=-1) 


def project(proj, pts):
    pts_hom = np.insert(pts, 3, 1)
    pts = proj @ pts_hom
    pts /= pts[-1]
    return pts[:2]

def gen_scale(new_w, new_h, old_w, old_h):
    scale_h = new_h / old_h
    scale_w = new_w / old_w
    scale = np.array([[scale_w,   0,    0],
                      [  0,   scale_h, 0 ],
                      [  0,      0,    1 ]])
    return scale, scale_h, scale_w

class MetricDict(defaultdict):
    def __init__(self):
        super().__init__(float)
        self.count = defaultdict(int)
    
    def __add__(self, other):
        for key, value in other.items():
            self[key] += value
            self.count[key] += 1
        return self
    @property
    def mean(self):
        return { key: self[key] / self.count[key] for key in self.keys() }

def to_numpy(data):
    if isinstance(data, np.ndarray):
        return data
    elif isinstance(data, torch.Tensor):
        return data.detach().cpu().detach().numpy()
    else:
        return np.array(data)

def grid_rot180(arr):
    if len(arr.shape) == 2:
        arr = arr[::-1, :]
        arr = arr[:, ::-1]
    elif len(arr.shape) == 3:
        arr = arr[:, ::-1, :]
        arr = arr[:, :, ::-1]
    return arr

def record(save_dir, content):
    if not os.path.exists(os.path.dirname(save_dir)):
        os.mkdir(os.path.dirname(save_dir))
        
    with open(save_dir, encoding='utf-8', mode='a') as f:
        f.write(content)

def scale_image_bboxes_kyps(batch_dict, scale_ratio=3):
    # scale image
    images = batch_dict['images']
    H, W = images.shape[-2:]
    # (bs, 3, h, w) -> (bs, 3, h//scale_ratio, w//scale_ratio) -> (bs, h//scale_ratio, w//scale_ratio, 3)
    batch_dict['images'] = F.interpolate(images, size=(H//scale_ratio, W//scale_ratio)).permute(0, 2, 3, 1)
    # scale bboxes
    bboxes_list = list()
    for bbox in batch_dict['pred_boxes']:
        bbox[:, :4] /= scale_ratio
        bboxes_list.append(bbox)
    batch_dict['pred_boxes'] = bboxes_list
    # scale keypoints
    kyps_list = list()
    for kyp in batch_dict['kyps_align']:
        kyp = kyp.squeeze(1) / scale_ratio
        kyps_list.append(kyp)
    batch_dict['kyps_align'] = kyps_list
    return batch_dict

