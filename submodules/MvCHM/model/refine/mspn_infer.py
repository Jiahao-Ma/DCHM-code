import os, sys
from unittest import result; sys.path.append(os.getcwd())
import argparse
from tqdm import tqdm
import numpy as np
import cv2
from PIL import Image

import torch
import torch.distributed as dist

import torchvision.transforms as transforms

import matplotlib.pyplot as plt

from model.refine.mspn_config import mspn_cfg as cfg
from model.refine.mspn_net import MSPN
from model.refine.mspn_data import get_affine_transform
from model.refine.mspn_attribute import load_dataset
from model.refine.mspn_data import Wildtrack
from model.refine.mspn_lib import get_logger

def affine_transform(pt, t):
    new_pt = np.array([pt[0], pt[1], 1.])
    new_pt = np.dot(t, new_pt)
    return new_pt[:2]

def get_test_loader(cfg, stage):
    # -------- get raw dataset interface -------- #
    normalize = transforms.Normalize(mean=cfg.INPUT.MEANS, std=cfg.INPUT.STDS)
    transform = transforms.Compose([transforms.ToTensor(), normalize])
    attr = load_dataset(cfg.DATASET.NAME)
    if cfg.DATASET.NAME == 'WILDTRACK':
        Dataset = Wildtrack
    return Dataset(attr, stage, transform)

def get_results(outputs, centers, scales, kernel=11, shifts=[0.25]):
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
            kps[w] = np.array([x * 4 + 2, (y+1) * 4 + 2]) # TODO: Add offset !
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
    
    return aligned_preds, preds, maxvals

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
    for img, gt, pred in zip(imgs, gt_kyps, pred_kyps):
        img = reverse(img)
        plt.imshow(img)
        plt.scatter(pred[0], pred[1], c='blue', label='pred')
        plt.scatter(gt[2], gt[3], c='green', label='gt') # gt: (kyp1_x, kyp1_y, kyp2_x, kyp2_y) kyp1 in 2D image, kyp2 in detection bbox
        print('x_diff: ', gt[2]-pred[0], '\t y_diff: ', gt[3]-pred[1])
        plt.legend()
        plt.axis('off')
        plt.show()


def compute_on_dataset(model, image_all_frames, scale_all_frames, cnt_all_frames, kyp_all_frames, device):
    model.eval()
    cpu_device = torch.device("cpu")
    pred_kyp_all_frames = list()
    for imgs, scales, centers, gt_kyps in zip(image_all_frames, scale_all_frames, cnt_all_frames, kyp_all_frames):
        imgs = imgs.to(device) # (n_bboxes, 3, h, w)
        with torch.no_grad():
            outputs = model(imgs)
            outputs = outputs.to(cpu_device).numpy()

        centers = np.array(centers)
        scales = np.array(scales)
        aligned_preds, preds, maxvals = get_results(outputs, centers, scales,
                cfg.TEST.GAUSSIAN_KERNEL, cfg.TEST.SHIFT_RATIOS)
        

        if True:
            visualize(imgs, gt_kyps, preds)

        # kp_scores = maxvals.squeeze(axis=-1).mean(axis=1)
        preds = np.concatenate((preds, maxvals), axis=2)

        results = list() 
        for i in range(preds.shape[0]):
            keypoints = preds[i].reshape(-1).tolist()

            results.append(dict(
                                keypoints=keypoints,
                                aligned_keypoints = aligned_preds[i].reshape(-1).tolist(),
                                ))
        pred_kyp_all_frames.append(results)                                

    return pred_kyp_all_frames


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument("--iter", "-i", type=int, default=-1)
    args = parser.parse_args()

    normalize = transforms.Normalize(mean=cfg.INPUT.MEANS, std=cfg.INPUT.STDS)
    transform = transforms.Compose([transforms.ToTensor(), normalize])

    num_gpus = int(
            os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    distributed =  num_gpus > 1

    if distributed:
        torch.cuda.set_device(args.local_rank)
        dist.init_process_group(backend="nccl", init_method="env://")
        # synchronize()

    # if is_main_process() and not os.path.exists(cfg.TEST_DIR):
    #     os.mkdir(cfg.TEST_DIR)
    logger = get_logger(
            cfg.DATASET.NAME, cfg.TEST_DIR, args.local_rank, 'test_log.txt')

    if args.iter == -1:
        logger.info("Please designate one iteration.")

    model = MSPN(cfg)
    device = torch.device(cfg.MODEL.DEVICE)
    model.to(cfg.MODEL.DEVICE)

    model_file = r'model\refine\checkpoint\mspn.pth'
    if os.path.exists(model_file):
        state_dict = torch.load(
                model_file, map_location=lambda storage, loc: storage)
        state_dict = state_dict['model']
        model.load_state_dict(state_dict)

    # data generate!
    data_loader = get_test_loader(cfg, 'train')
    per_frame_infos = data_loader.gen_data_per_frame()
    image_all_frames = list()
    cnt_all_frames = list()
    scale_all_frames = list()
    kyp_all_frames = list()
    for bboxes, img_path in per_frame_infos:
        data_numpy = cv2.imread(img_path, cv2.IMREAD_COLOR)
        data_numpy = cv2.cvtColor(data_numpy, cv2.COLOR_BGR2RGB)
        # ax = plt.figure().add_subplot(111)
        # for bbox in bboxes:
        #     bb = bbox['bbox']
        #     kyp = bbox['joints']
        #     rect = plt.Rectangle(bb[:2], bb[2], bb[3], fill=None)
        #     ax.scatter(kyp[0], kyp[1], c='red')
        #     ax.add_patch(rect)
        # ax.imshow(data_numpy)
        # plt.show()
        image_per_frames = list()
        cnt_per_frames = list()
        scale_per_frames = list()
        kyp_per_frames = list()
        for bbox in bboxes:
            kyp = np.zeros((4))
            bb = bbox['bbox']
            kyp[:2] = bbox['joints'][:2] # store keypoints in 2D image
            center = bbox['center'] 
            scale = bbox['scale']
            rotation = 0
            scale[0] *= (1 + 0.2)
            scale[1] *= (1 + 0.2)
            # fit the ratio
            if scale[0] > data_loader.w_h_ratio * scale[1]:
                scale[1] = scale[0] * 1.0 / data_loader.w_h_ratio
            else:
                scale[0] = scale[1] * 1.0 * data_loader.w_h_ratio
            trans = get_affine_transform(center, scale, rotation, data_loader.input_shape)
            kyp[2:] = affine_transform(bbox['joints'][:2], trans) 
            
            img = cv2.warpAffine(
                    data_numpy,
                    trans,
                    (int(data_loader.input_shape[1]), int(data_loader.input_shape[0])),
                    flags=cv2.INTER_LINEAR)
            img = transform(img)

            image_per_frames.append(img)
            kyp_per_frames.append(kyp)
            cnt_per_frames.append(center)
            scale_per_frames.append(scale)
            
        image_per_frames = torch.stack(image_per_frames, dim=0) # (n_frames, 3, h, w)
        kyp_per_frames = np.stack(kyp_per_frames, axis=0) # (n_frames, 3, h, w)
        cnt_per_frames = np.stack(cnt_per_frames, axis=0) # (n_frames, 3, h, w)
        scale_per_frames = np.stack(scale_per_frames, axis=0) # (n_frames, 3, h, w)

        image_all_frames.append(image_per_frames)
        cnt_all_frames.append(cnt_per_frames)
        scale_all_frames.append(scale_per_frames)
        kyp_all_frames.append(kyp_per_frames)
    
    # Forward !
    results = compute_on_dataset(model, image_all_frames, scale_all_frames, cnt_all_frames, kyp_all_frames, device)

    # Visualization !
    for cam, (bboxes, img_path) in enumerate(per_frame_infos):
        img = Image.open(img_path)
        pred_kyps = results[cam]
        ax = plt.figure(figsize=(15, 8)).add_subplot(111)
        # visualize pred
        for pred_kyp in pred_kyps:
            ax.scatter(x=pred_kyp['aligned_keypoints'][0], 
                       y=pred_kyp['aligned_keypoints'][1], 
                       label='pred', color='blue')
        # visualize gt
        ax.scatter(x = kyp_all_frames[cam][:, 0], 
                   y = kyp_all_frames[cam][:, 1], 
                   label='gt', color='green')
        ax.imshow(img)
        ax.legend()
        plt.show()
        


if __name__ == '__main__':
    main()