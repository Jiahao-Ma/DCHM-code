import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import cv2
import json
import argparse
import torch
import tqdm
import numpy as np
import matplotlib.pyplot as plt
from dataset.wildtrack import Wildtrack
from torch.utils.data import DataLoader
from skimage.segmentation import slic, mark_boundaries
from skimage.color import label2rgb
from skimage.measure import regionprops
from concurrent.futures import ThreadPoolExecutor, as_completed
from glob import glob
# First round initialization -> make superpixel for foreground mask
def mask_superpixel(data, n_segments=-1):
    
    images = data['nonnorm_image'].squeeze(0).cuda()
    masks_split = data['mask'].squeeze(0)
    
    compactness = 20
    sigma = 0.5
    all_centroids = []
    all_centroids_idx = []
    all_images = []
    for i in range(images.shape[0]):
        img = images[i].permute(1, 2, 0) * 255
        masks = masks_split[i].to(dtype=torch.int32)
        masked_img = torch.zeros_like(img)
        group_centroids = []
        centroids_idx = []
        for mask_id in range(int(masks.max()+1)):
            if mask_id == 0:
                continue   # 0 is background by default
            mask = masks == mask_id
            if n_segments <= 0:
                masked_img[mask] = torch.mean(img[mask], dim=0)
                group_centroids.append(torch.mean(torch.where(mask), dim=1).to(dtype=torch.int32))
            else:
                each_mask_img = torch.zeros_like(img)  
                if img[mask].shape[0] <= n_segments:
                    continue
                each_mask_img[mask] = img[mask]
                # TODO: change to opencv slic
                segments = slic(each_mask_img.cpu().numpy(), n_segments=n_segments, compactness=compactness, sigma=sigma,
                                start_label=1, mask=mask.cpu().numpy(), enforce_connectivity=True)
                color_segments = label2rgb(segments, each_mask_img.cpu().numpy(), kind='avg', bg_label=0)
                color_segments = torch.from_numpy(color_segments).to(device=masked_img.device)
                regions = regionprops(segments)
                centroids = [region.centroid for region in regions]
                centroids = torch.Tensor([cnt for cnt in centroids if mask[int(cnt[0]), int(cnt[1])]])
                if len(centroids) == 0:
                    masked_img[mask] = torch.mean(img[mask], dim=0)
                    group_centroids.append(torch.mean(torch.where(mask), dim=1).to(dtype=torch.int32).reshape(1, 2))
                else:
                    masked_img[mask] = color_segments[mask]
                    group_centroids.append(centroids)
                centroids_idx.append(torch.ones(len(group_centroids[-1])) * mask_id)
        group_centroids = torch.cat(group_centroids, dim=0).contiguous().view(-1, 2).cpu().numpy()
        centroids_idx = torch.cat(centroids_idx, dim=0).contiguous().cpu().numpy()
        all_centroids.append(group_centroids)
        all_centroids_idx.append(centroids_idx)
        all_images.append(masked_img.cpu().numpy().astype(np.uint8))
    return all_centroids, all_centroids_idx, all_images

# Function to process a single mask for a given image
def process_single_mask(img, mask_id, masks, n_segments, compactness, sigma):
    masked_img = torch.zeros_like(img)
    group_centroids = []
    centroids_idx = []

    mask_area = masks == mask_id
    if mask_area.sum() == 0:  # If no area belongs to this mask, skip
        return None

    if n_segments <= 0:
        masked_img[mask_area] = torch.mean(img[mask_area], dim=0)
        group_centroids.append(torch.mean(torch.where(mask_area), dim=1).to(dtype=torch.int32))
    else:
        each_mask_img = torch.zeros_like(img)
        if img[mask_area].shape[0] <= n_segments:
            return None

        each_mask_img[mask_area] = img[mask_area]

        # Perform SLIC segmentation
        segments = slic(each_mask_img.cpu().numpy(), n_segments=n_segments, compactness=compactness, sigma=sigma,
                        start_label=1, mask=mask_area.cpu().numpy(), enforce_connectivity=True)
        
        color_segments = label2rgb(segments, each_mask_img.cpu().numpy(), kind='avg', bg_label=0)
        color_segments = torch.from_numpy(color_segments).to(device=masked_img.device)

        regions = regionprops(segments)
        centroids = [region.centroid for region in regions]
        centroids = torch.Tensor([cnt for cnt in centroids if mask_area[int(cnt[0]), int(cnt[1])]])

        if len(centroids) == 0:
            masked_img[mask_area] = torch.mean(img[mask_area], dim=0)
            group_centroids.append(torch.mean(torch.where(mask_area), dim=1).to(dtype=torch.int32).reshape(1, 2))
        else:
            masked_img[mask_area] = color_segments[mask_area]
            group_centroids.append(centroids)
        centroids_idx.append(torch.ones(len(group_centroids[-1])) * mask_id)

    return torch.cat(group_centroids, dim=0).contiguous().view(-1, 2), torch.cat(centroids_idx, dim=0), masked_img.cpu().numpy().astype(np.uint8)


# Main function that uses mask-level parallelism
def mask_superpixel_mt(data, n_segments=-1, num_threads=4, bg_color='white'):
    
    images = data['nonnorm_image'].squeeze(0).cuda()
    masks_split = data['mask'].squeeze(0)

    compactness = 20
    sigma = 0.5
    all_centroids = []
    all_centroids_idx = []
    all_images = []
    
    for i in range(images.shape[0]):
        img = images[i].permute(1, 2, 0) #* 255
        masks = masks_split[i]
        masked_img = torch.zeros_like(img)
        group_centroids = []
        centroids_idx = []

        # Use ThreadPoolExecutor to parallelize mask processing
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = []

            for mask_id in range(1, int(masks.max()) + 1):  # Start from 1 (skip background)
                # Submit each mask to be processed in a separate thread
                futures.append(executor.submit(process_single_mask, img, mask_id, masks, n_segments, compactness, sigma))
            
            for future in as_completed(futures):
                result = future.result()
                if result is not None:
                    group_centroids_result, centroids_idx_result, masked_img_result = result
                    group_centroids.append(group_centroids_result)
                    centroids_idx.append(centroids_idx_result)
                    masked_img += torch.from_numpy(masked_img_result).to(masked_img.device)

        if bg_color == 'white':
            # if bg_color is white (rgb = (0, 0, 0)), set the background to white
            mask = torch.all(masked_img == 0, dim=-1).to(device=masked_img.device)
            masked_img[mask] = torch.tensor([255, 255, 255], dtype=masked_img.dtype, device=masked_img.device)
            
        if len(group_centroids) > 0:
            group_centroids = torch.cat(group_centroids, dim=0).cpu().numpy()
            centroids_idx = torch.cat(centroids_idx, dim=0).cpu().numpy()
        else:
            group_centroids = np.zeros((0, 2))
            centroids_idx = np.zeros(0)
        all_centroids.append(group_centroids)
        all_centroids_idx.append(centroids_idx)
        all_images.append(masked_img.cpu().numpy().astype(np.uint8))
    return all_centroids, all_centroids_idx, all_images

if __name__ == '__main__':
    target = 'people' # 'people by default
    parser = argparse.ArgumentParser("Grounded-Segment-Anything Demo", add_help=True)
    parser.add_argument("--root", type=str, required=True, help="path to image folder")
    parser.add_argument("--start_with", type=int, default=0, help="The index of the first image to start with")
    parser.add_argument("--end_with", type=int, default=-1, help="The index of the last image to end with")
    parser.add_argument("--n_segments", type=str, required=True, help="The number of segments for slic, e.g., '30,60'")

    args_str = [
    "--root", "/home/jiahao/Downloads/data/wildtrack_data_gt",
    "--start_with", "360",
    "--end_with", "361",
    "--n_segments", "60"
    ]

    args = parser.parse_args(args_str) # FOR DEBUG
    # args = parser.parse_args()
    n_segments = [int(x) for x in args.n_segments.split(',')]
    
    output_dir = os.path.join(args.root, 'superpixels')
    dataset = Wildtrack(args.root, mask_type=target, mask_label_type='split', start_with=args.start_with, end_with=args.end_with)
    dataloader = DataLoader(dataset, batch_size=1, pin_memory=True, num_workers=4, drop_last=True)
    
    pbar = tqdm.tqdm(dataloader, total=len(dataloader))
    for data in pbar:
        for n_segment in n_segments:
            save_dir = os.path.join(output_dir, data['frame_idx'][0], str(n_segment))
            # if os.path.exists(os.path.join(save_dir, 'centroids.json')):
            #     print(f"{os.path.join(save_dir, 'centroids.json')} exists, skip.")
            #     continue
            pbar.set_description("Processing {}".format(data['frame_idx'][0]))
            pbar.refresh()
            # all_centroids, all_centroids_idx, all_images = mask_superpixel(data, n_segment) # single thread (give up)
            all_centroids, all_centroids_idx, all_images = mask_superpixel_mt(data, n_segment, num_threads=8) # multi thread
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            for cam_idx, image in enumerate(all_images):
                cv2.imwrite(os.path.join(save_dir, 'cam{}.png'.format(cam_idx+1)), image[...,::-1])
            # save all centroids and centroids_idx using json, make them together
            centroids = {'centroids': [cnt.tolist() for cnt in all_centroids], 
                         'centroids_idx': [idx.tolist() for idx in all_centroids_idx]}
            # save in json
            with open(os.path.join(save_dir, 'centroids.json'), 'w') as f:
                json.dump(centroids, f)
            
    print('All done!')
    