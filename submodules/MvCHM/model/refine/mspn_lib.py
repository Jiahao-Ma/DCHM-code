import logging
import coloredlogs
import os
import cv2
import math

import torch
import numpy as np
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from lib.utils.visual_utils import Process
from lib.utils.tool_utils import to_numpy
from model.refine.mspn_attribute import load_dataset
from model.refine.mspn_data import Wildtrack

# todo remove color when logging in file
def get_logger(name='', save_dir=None, distributed_rank=0, filename="log.txt"):
    logger = logging.getLogger(name)
    coloredlogs.install(level='DEBUG', logger=logger)
    # logger.setLevel(logging.DEBUG)
    # don't log results for the non-master process
    if distributed_rank > 0:
        return logger
    formatter = logging.Formatter(
        "%(asctime)s %(name)s %(levelname)s: %(message)s")

    # ch = logging.StreamHandler(stream=sys.stdout)
    # ch.setLevel(logging.DEBUG)
    # ch.setFormatter(formatter)
    # logger.addHandler(ch)
    if save_dir:
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        fh = logging.FileHandler(os.path.join(save_dir, filename))
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    return logger

def get_test_loader(cfg, num_gpu, local_rank, stage, is_dist=True):
    # -------- get raw dataset interface -------- #
    normalize = transforms.Normalize(mean=cfg.INPUT.MEANS, std=cfg.INPUT.STDS)
    transform = transforms.Compose([transforms.ToTensor(), normalize])
    attr = load_dataset(cfg.DATASET.NAME)
   
    if cfg.DATASET.NAME == 'WILDTRACK':
        Dataset = Wildtrack
    dataset = Dataset(attr, stage, transform)

    # -------- split dataset to gpus -------- #
    num_data = dataset.__len__()
    num_data_per_gpu = math.ceil(num_data / num_gpu)
    st = local_rank * num_data_per_gpu
    ed = min(num_data, st + num_data_per_gpu)
    indices = range(st, ed)
    subset= torch.utils.data.Subset(dataset, indices)

    # -------- make samplers -------- #
    sampler = torch.utils.data.sampler.SequentialSampler(subset)

    images_per_gpu = cfg.TEST.IMS_PER_GPU

    batch_sampler = torch.utils.data.sampler.BatchSampler(
            sampler, images_per_gpu, drop_last=False)

    data_loader = DataLoader(
            subset, num_workers=cfg.DATALOADER.NUM_WORKERS,
            batch_sampler=batch_sampler,)
            
    data_loader.ori_dataset = dataset

    return data_loader

def get_dir(src_point, rot_rad):
    sn, cs = np.sin(rot_rad), np.cos(rot_rad)

    src_result = [0, 0]
    src_result[0] = src_point[0] * cs - src_point[1] * sn
    src_result[1] = src_point[0] * sn + src_point[1] * cs

    return src_result

def get_3rd_point(a, b):
    direct = a - b
    return b + np.array([-direct[1], direct[0]], dtype=np.float32)

def get_affine_transform(center, scale, rot, output_size):
    if not isinstance(scale, np.ndarray) and not isinstance(scale, list):
        scale = np.array([scale, scale])
    scale_tmp = scale * 200.0

    src_w = scale_tmp[0]
    dst_w = output_size[1]
    dst_h = output_size[0]

    rot_rad = np.pi * rot / 180
    src_dir = get_dir([0, src_w * -0.5], rot_rad)
    dst_dir = np.array([0, dst_w * -0.5], np.float32)

    src = np.zeros((3, 2), dtype=np.float32)
    dst = np.zeros((3, 2), dtype=np.float32)
    src[0, :] = center
    src[1, :] = center + src_dir
    dst[0, :] = [dst_w * 0.5, dst_h * 0.5]
    dst[1, :] = np.array([dst_w * 0.5, dst_h * 0.5]) + dst_dir

    src[2:, :] = get_3rd_point(src[0, :], src[1, :])
    dst[2:, :] = get_3rd_point(dst[0, :], dst[1, :])

    trans = cv2.getAffineTransform(np.float32(src), np.float32(dst))

    return trans


def affine_transform(pt, t):
    new_pt = np.array([pt[0], pt[1], 1.])
    new_pt = np.dot(t, new_pt)
    return new_pt[:2]


def mspn_preprocess(batch_dict, cfg, device, process:Process, pixel_std=200):
    def _bbox_to_center_and_scale(bbox, pixel_std=200):
        x, y, w, h = bbox

        center = np.zeros(2, dtype=np.float32)
        center[0] = x + w / 2.0
        center[1] = y + h / 2.0

        scale = np.array([w * 1.0 / pixel_std, h * 1.0 / pixel_std],
                dtype=np.float32)

        return center, scale
    batch_dict['cropped_images'] = list()
    batch_dict['centers'] = list()
    batch_dict['scales'] = list()
    # random move center
    # bboxes = list()
    # for bbox in batch_dict['pred_boxes']:
    #     rand_off_w = np.random.randint(0, 30, size=(bbox.shape[0], 1))
    #     rand_off_h = np.random.randint(0, 30, size=(bbox.shape[0], 1))
    #     bbox[:, [0, 2]] += rand_off_w
    #     bbox[:, [1, 3]] += rand_off_h
    #     bboxes.append(bbox)
    # batch_dict['pred_boxes'] = bboxes
    for cam, bboxes in enumerate(batch_dict['pred_boxes']): # for each frame
        image = batch_dict['images'][cam]
        images, centers, scales = list(), list(), list()
        for bbox in bboxes: # for each group of bboxes
            if bbox[0] == bbox[1] == bbox[2] == bbox[3]:
                continue
            x1, y1, x2, y2 = bbox[:4]
            w = x2 - x1
            h = y2 - y1
            bb = np.array([x1, y1, w, h])
            center, scale = _bbox_to_center_and_scale(bb, pixel_std)
            scale[0] *= (1 + cfg.BBOX_X_EXTENSION)
            scale[1] *= (1 + cfg.BBOX_Y_EXTENSION)
            # scale[0] *= (1 + np.random.uniform(0, 0.5))
            # scale[1] *= (1 + np.random.uniform(0, 0.5))
        
            # fit the ratio
            if scale[0] > cfg.WIDTH_HEIGHT_RATIO * scale[1]:
                scale[1] = scale[0] * 1.0 / cfg.WIDTH_HEIGHT_RATIO
            else:
                scale[0] = scale[1] * 1.0 * cfg.WIDTH_HEIGHT_RATIO
            
            rotation = 0
            trans = get_affine_transform(center, scale, rotation, cfg.INPUT_SHAPE)

            img = cv2.warpAffine(image,
                                 trans,
                                 (int(cfg.INPUT_SHAPE[1]), int(cfg.INPUT_SHAPE[0])),
                                 flags=cv2.INTER_LINEAR)
            
            img = process.normalize_image_PIL(img)
            images.append(img)
            centers.append(center)
            scales.append(scale)
        images = torch.stack(images, dim=0) 
        centers = np.array(centers)
        scales = np.array(scales)
        batch_dict['cropped_images'].append(images.to(device=device))
        batch_dict['centers'].append(centers)
        batch_dict['scales'].append(scales)
    return batch_dict

def mspn_postprocess(batch_dict, cfg, kernel=11, shifts=[0.25]):
    batch_dict['kyps'] = list()
    batch_dict['kyps_align'] = list()
    for idx, (outputs, centers, scales) in enumerate(zip(batch_dict['keypoints'], batch_dict['centers'], batch_dict['scales'])):
        outputs = to_numpy(outputs)
        scales *= 200
        nr_img = outputs.shape[0]
        aligned_preds = np.zeros((nr_img, cfg.DATASET.KEYPOINT.NUM, 2))
        preds = np.zeros((nr_img, cfg.DATASET.KEYPOINT.NUM, 2))
        maxvals = np.zeros((nr_img, cfg.DATASET.KEYPOINT.NUM, 1))
        for i in range(nr_img):
            score_map = outputs[i].copy()
            score_map = score_map / 255 + 0.5
            kps = np.zeros((cfg.DATASET.KEYPOINT.NUM, 2))
            scores = np.zeros((cfg.DATASET.KEYPOINT.NUM, 1))
            border = 10
            dr = np.zeros((cfg.DATASET.KEYPOINT.NUM,
                cfg.OUTPUT_SHAPE[0] + 2 * border, cfg.OUTPUT_SHAPE[1] + 2 * border))
            dr[:, border: -border, border: -border] = outputs[i].copy()
            for w in range(cfg.DATASET.KEYPOINT.NUM):
                dr[w] = cv2.GaussianBlur(dr[w], (kernel, kernel), 0)
            for w in range(cfg.DATASET.KEYPOINT.NUM):
                for j in range(len(shifts)):
                    if j == 0:
                        lb = dr[w].argmax()
                        y, x = np.unravel_index(lb, dr[w].shape)
                        dr[w, y, x] = 0
                        x -= border
                        y -= border
                    lb = dr[w].argmax()
                    py, px = np.unravel_index(lb, dr[w].shape)
                    dr[w, py, px] = 0
                    px -= border + x
                    py -= border + y
                    ln = (px ** 2 + py ** 2) ** 0.5
                    if ln > 1e-3:
                        x += shifts[j] * px / ln
                        y += shifts[j] * py / ln
                x = max(0, min(x, cfg.OUTPUT_SHAPE[1] - 1))
                y = max(0, min(y, cfg.OUTPUT_SHAPE[0] - 1))
                kps[w] = np.array([x * 4 + 2, (y) * 4 + 2]) # TODO: Add offset !
                scores[w, 0] = score_map[w, int(round(y) + 1e-9), \
                        int(round(x) + 1e-9)]
            # aligned or not ...
            aligned_kps = np.zeros_like(kps)
            aligned_kps[:, 0] = kps[:, 0] / cfg.INPUT_SHAPE[1] * scales[i][0] + \
                    centers[i][0] - scales[i][0] * 0.5
            aligned_kps[:, 1] = kps[:, 1] / cfg.INPUT_SHAPE[0] * scales[i][1] + \
                    centers[i][1] - scales[i][1] * 0.5 
            aligned_preds[i] = aligned_kps
            preds[i] = kps
            maxvals[i] = scores 
        # visualize(batch_dict['cropped_images'][idx], None, preds)
        batch_dict['kyps'].append(preds)
        batch_dict['kyps_align'].append(aligned_preds)
    
    return batch_dict


def visualize(imgs, gt_kyps, pred_kyps):
    if len(pred_kyps.shape) == 3:
        pred_kyps = pred_kyps.squeeze(1)
    def reverse(img):
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img = img.permute(1,2,0).cpu().detach().numpy() # (h, w, c)
        img = img * std + mean
        img = (img * 255).astype(np.uint8)
        return img
    for img, pred in zip(imgs, pred_kyps):
        img = reverse(img)
        plt.imshow(img)
        plt.scatter(pred[0], pred[1], c='blue', label='pred')
        # plt.scatter(gt[2], gt[3], c='green', label='gt') # gt: (kyp1_x, kyp1_y, kyp2_x, kyp2_y) kyp1 in 2D image, kyp2 in detection bbox
        # print('x_diff: ', gt[2]-pred[0], '\t y_diff: ', gt[3]-pred[1])
        plt.legend()
        plt.axis('off')
        plt.show()
