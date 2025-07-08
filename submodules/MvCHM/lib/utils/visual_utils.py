import cv2
import torch
import numpy as np
import torch.nn.functional as F
from torchvision import transforms

import matplotlib.pyplot as plt
from lib.data.dataloader_bk import UnNormalizer
from lib.data.wildtrack import Wildtrack
from lib.data.multiviewX import MultiviewX
from lib.utils import tool_utils

color = {'green':(0,255,0),
        'blue':(255,165,0),
        'dark red':(0,0,139),
        'red':(0, 0, 255),
        'dark slate blue':(139,61,72),
        'aqua':(255,255,0),
        'brown':(42,42,165),
        'deep pink':(147,20,255),
        'fuchisia':(255,0,255),
        'yello':(0,238,238),
        'orange':(0,165,255),
        'saddle brown':(19,69,139),
        'black':(0,0,0),
        'white':(255,255,255)}

def save_image(image, save_dir):
    plt.figure(figsize=(15, 8))
    plt.axis('off')
    plt.imshow(image)
    plt.savefig(save_dir, bbox_inches='tight', pad_inches=0, dpi=300)
    print('Image has been saved in ', save_dir)
    plt.close()

def draw_boxes(img, boxes, pred_keypoints=None, gt_keypoints=None, scores=None, tags=None, line_thick=1, line_color='white', pred_kp_color='yellow', gt_kp_color='purple', ax=None, mask=None):
    if ax is None:
        fig = plt.figure(figsize=(15, 8))
        ax = plt.subplot(111)
        ax.imshow(img)
        ax.axis('off')
    else:
        ax.imshow(img)
    width = img.shape[1]
    height = img.shape[0]
    for i in range(len(boxes)):
        one_box = boxes[i]
        one_box = np.array([max(one_box[0], 0), max(one_box[1], 0),
                    min(one_box[2], width - 1), min(one_box[3], height - 1)])
        x1,y1,x2,y2 = np.array(one_box[:4]).astype(int)
        if mask is not None:
            if mask[i]: # within range
                color = 'green'
            else: 
                color = 'red'
        else:
            color = 'red'
        rect = plt.Rectangle([x1, y1], width=x2-x1, height=y2-y1, fill=False, color=color, linewidth=1.5)
        ax.add_patch(rect)
        if scores is not None:
            text = "{:.3f}".format(scores[i])
            ax.text(x1, y1-7, text, color='yellow', fontsize=4)
    if pred_keypoints is not None:
        for kp in pred_keypoints:
            if kp[0] < 0 or kp[0] > width or kp[1] < 0 or kp[1] > height:
                continue
            ax.scatter(kp[0], kp[1], c=pred_kp_color, s=5)
    if gt_keypoints is not None:
        for kp in gt_keypoints:
            if kp[0] < 0 or kp[0] > width or kp[1] < 0 or kp[1] > height:
                continue
            ax.scatter(kp[0], kp[1], c=gt_kp_color, s=5)
    
    return ax


def project(pts, proj):
    pts = pts.T
    pts = proj @ pts
    pts = pts / pts[-1, :]
    return pts[:2, :].T

def reverse_image(img):
    # image_mean = np.array([103.530, 116.280, 123.675])
    # image_std = np.array([57.375, 57.120, 58.395])
    # mean = torch.tensor(image_mean/255, dtype=torch.float32)
    # std = torch.tensor(image_std/255, dtype=torch.float32)
    mean = torch.tensor(np.array([0.485, 0.456, 0.406]), dtype=torch.float32)
    std = torch.tensor(np.array([0.229, 0.224, 0.225]), dtype=torch.float32)
    unnormalize = UnNormalizer(mean=mean, std=std)

    img = np.array(255 * unnormalize(img))
    img = np.clip(img, a_min=0, a_max=255)
    img = np.transpose(img.astype(np.uint8), (1, 2, 0))
    return img


def visualize_kyp_heatmap(pred, gt, kyp_map):
    fig = plt.figure(num='heatmap', figsize=(15, 8))
    fig.clear()

    _format_heatmap(pred, ax=plt.subplot(131))
    _format_heatmap(kyp_map, ax=plt.subplot(132))
    _format_heatmap(gt, ax=plt.subplot(133))
    
    return fig      

def visualize_heatmap(pred, gt):
    fig = plt.figure(num='heatmap', figsize=(15, 8))
    fig.clear()

    _format_heatmap(pred, ax=plt.subplot(121))
    _format_heatmap(gt, ax=plt.subplot(122))
    
    return fig      
    

def _format_heatmap(heatmap, ax=None):
    if isinstance(heatmap, torch.Tensor):
        heatmap = (heatmap.detach().cpu().numpy() * 255).astype(np.uint8)
    else:
        heatmap = (heatmap * 255).astype(np.uint8)

    # Create a new axis if one is not provided
    if ax is None:
        fig = plt.figure()
        ax = fig.gca()
    
    # Plot scores
    ax.clear()
    ax.imshow(heatmap)

    # Format axes
    ax.set_xlabel('x')
    ax.set_ylabel('y')

    return ax, heatmap

def grid_rot180(arr):
    if len(arr.shape) == 2:
        arr = arr[::-1, :]
        arr = arr[:, ::-1]
    elif len(arr.shape) == 3:
        arr = arr[:, ::-1, :]
        arr = arr[:, :, ::-1]
    return arr


class Monitor(object):
    def __init__(self) -> None:
        pass

    def visualize(self, batch_dict, batch_pred, batch_gt, show=False):
        fig = plt.figure(figsize=(15,8))
        axes= fig.subplots(3, 1)
        axes = axes.reshape(-1)
        # ---- feature_map ---- #
        gt_heatmap = batch_gt['heatmap'].squeeze()
        bev_features = batch_dict['spatial_features']
        pred_features = batch_pred['heatmap'] 
        bev_features = torch.norm(torch.norm(bev_features, dim=0), dim=0)
        pred_features = torch.norm(torch.norm(pred_features, dim=-1), dim=0)
        axes[0], gt_heatmap = _format_heatmap(gt_heatmap, axes[0])
        axes[1], bev_features = _format_heatmap(bev_features, axes[1])
        axes[2], pred_features = _format_heatmap(pred_features, axes[2])
  
        axes=axes.reshape(3, 1)
        fig.tight_layout()
        if show:
            plt.show()
        else:
            return fig, gt_heatmap, bev_features
    
    def viz_gt_bbox(self, batch_gt, show=False, config=None, dataset=None, viz_grid=True):
        if viz_grid:
            assert config is not None and dataset is not None
            worldgrid = tool_utils.make_grid(world_size=config.DATA_CONFIG.GRID_RANGE[-3:-1],
                                             cube_LW=config.DATA_CONFIG.VOXEL_SIZE[:-1],
                                             dataset=config.DATA_CONFIG.DATASET)
            worldcoords = Wildtrack.get_worldcoord_from_worldgrid(worldgrid.permute(2,0,1).cpu().numpy())
            worldcoords = worldcoords.reshape(3,-1).T
            worldcoords_hom = np.ones(shape=(worldcoords.shape[0], 4))
            worldcoords_hom[:, :3] = worldcoords

        fig = plt.figure(figsize=(15,12))
        imgs = batch_gt['img']
        num_cam = imgs.shape[0]
        batch_annots = batch_gt['annot']
        batch_poses = batch_gt['pos']
        axes= fig.subplots(3, 3)
        axes = axes.reshape(-1)
        for cam in range(num_cam):
            if cam <= 5:
                axes_id = cam
            else:
                axes_id = cam + 1
            img = reverse_image(imgs[cam])
            annots = batch_annots[cam]
            poses = batch_poses[cam]
            for (bbox, pos) in zip(annots, poses):
                if bbox[0] == bbox[1] == bbox[2] == bbox[3] == -1.0:
                    continue
                bbox = np.array(bbox)
                bbox[0] = np.maximum(bbox[0], 0)
                bbox[1] = np.maximum(bbox[1], 0)
                bbox[2] = np.minimum(bbox[2], img.shape[1])
                bbox[3] = np.minimum(bbox[3], img.shape[0])
                height = int(bbox[3]) - int(bbox[1])
                width = int(bbox[2]) - int(bbox[0])
                xy = bbox[:2]
                rect = plt.Rectangle(xy, width, height, fill=False, color='red', linewidth=1.5)
                axes[axes_id].add_patch(rect) 
                feet_pts = bbox[-2:]
                # axes[axes_id].scatter(coords[:, 0], coords[:, 1], s=0.5, c='green', alpha=0.3)
                axes[axes_id].scatter(x=feet_pts[0], y=feet_pts[1], s=10, c='yellow')
            axes[axes_id].imshow(img)
        for i in range(len(axes)):
            axes[i].axis('off')
        axes=axes.reshape(3, 3)
        fig.tight_layout()
        if show:
            plt.show()
        else:
            return fig
    
    def viz_pred_bbox(self, batch_dict, show=False):
        num_cam = len(batch_dict['pred_boxes'])
        fig = plt.figure(figsize=(15,12))
        axes= fig.subplots(3, 3)
        axes = axes.reshape(-1)
        for cam in range(num_cam):
            if cam <= 5:
                axes_id = cam
            else:
                axes_id = cam + 1
            axes[axes_id] = draw_boxes(img=reverse_image(batch_dict['images'][cam].cpu()),
                                        boxes=batch_dict['pred_boxes'][cam][:, :4],
                                        pred_keypoints = tool_utils.to_numpy(batch_dict['feet_pos'][cam]),
                                        gt_keypoints=tool_utils.to_numpy(batch_dict['annots'][cam][:, -2:]),
                                        scores=batch_dict['pred_boxes'][cam][:, 4],
                                        tags=['Person']*len(batch_dict['pred_boxes'][cam]),
                                        line_thick=1, line_color='black', ax=axes[axes_id])
    
        for i in range(len(axes)):
            axes[i].axis('off')
        axes=axes.reshape(3, 3)
        fig.tight_layout()
        if show:
            plt.show()
        else:
            return fig
        
    def viz_comb_pred_gt(self, batch_dict, show=False):
        num_cam = len(batch_dict['pred_boxes'])
        keypoints_samples = batch_dict['training_keypoint_samples']
        fig = plt.figure(figsize=(15,12))
        axes= fig.subplots(3, 3)
        axes = axes.reshape(-1)
        for cam in range(num_cam):
            if cam <= 5:
                axes_id = cam
            else:
                axes_id = cam + 1
            axes[axes_id] = draw_boxes(img=reverse_image(batch_dict['images'][cam].cpu()),
                                       boxes=tool_utils.to_numpy(keypoints_samples['ex_rects_list'][cam][:, :4]),
                                       scores=tool_utils.to_numpy(keypoints_samples['ex_rects_list'][cam][:, 4]),
                                       keypoints=tool_utils.to_numpy(keypoints_samples['keypoints_in_img_list'][cam]),
                                       tags=['Person']*len(batch_dict['pred_boxes'][cam]),
                                       line_thick=1, line_color='black', ax=axes[axes_id])
        
        for i in range(len(axes)):
            axes[i].axis('off')
        axes=axes.reshape(3, 3)
        fig.tight_layout()
        if show:
            plt.show()
        else:
            return fig


class Process(object):
    def __init__(self, scale_h, scale_w, pad_h, pad_w, new_h, new_w, old_h, old_w) -> None:
        self.scale_h = scale_h
        self.scale_w = scale_w
        self.pad_h = pad_h
        self.pad_w = pad_w
        self.new_h = new_h
        self.new_w = new_w
        self.old_h = old_h
        self.old_w = old_w

        self.mean = torch.tensor(np.array([0.485, 0.456, 0.406]), dtype=torch.float32)
        self.std = torch.tensor(np.array([0.229, 0.224, 0.225]), dtype=torch.float32)
        self.unnormalize = UnNormalizer(mean=self.mean, std=self.std)

        self._norm = transforms.Normalize(mean=self.mean, std=self.std)
        self.normalize = transforms.Compose([transforms.ToTensor(),
                                            self._norm
                                            ])
    
    def reverse_bbox(self, bboxes):
        '''
            Reverse the bbox to original size
        '''
        reverse_bboxes = list()
        for i in range(len(bboxes)):
                bboxes[i][:, 0] = np.maximum(bboxes[i][:, 0], 0)
                bboxes[i][:, 1] = np.maximum(bboxes[i][:, 1], 0)
                bboxes[i][:, 2] = np.minimum(bboxes[i][:, 2], self.new_w - self.pad_w) # W
                bboxes[i][:, 3] = np.minimum(bboxes[i][:, 3], self.new_h - self.pad_h) # H

                bboxes[i][:, [0, 2]] /= self.scale_w
                bboxes[i][:, [1, 3]] /= self.scale_h
                reverse_bboxes.append(bboxes[i])
        return reverse_bboxes
    
    def reverse_image_size(self, images):
        '''
            Reverse the new image to original image
        '''
        images_no_pad = torch.zeros((images.shape[0], images.shape[1], self.new_h - self.pad_h, self.new_w - self.pad_w))
        images_no_pad = images[:, :, :self.new_h - self.pad_h, :self.new_w - self.pad_w]
        return F.interpolate(images_no_pad, (self.old_h, self.old_w))

    def reverse_image_feat(self, img):
        img = np.array(255 * tool_utils.to_numpy(self.unnormalize(img))).astype(np.uint8)
        img = np.clip(img, a_min=0, a_max=255)
        img = np.transpose(img, (0, 2, 3, 1))
        return img
    
    def normalize_image_PIL(self, img):
        return self.normalize(img)
    
    def normalize_image_array(self, imgs):
        imgs = torch.stack([ torch.from_numpy(img).to(dtype=torch.float32) for img in imgs], dim=0).permute(0, 3, 1, 2)
        imgs = self._norm(imgs / 255) # (cam_num, 3, h, w)
        return imgs