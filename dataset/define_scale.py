import os,sys;sys.path.append(os.getcwd())
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2

from dataset.wildtrack import Wildtrack

def define_ground_range():
    image_idx = 0
    dataset = Wildtrack('/home/jiahao/Downloads/data/Wildtrack')
    xi = np.arange(40, 480, 10)
    yi = np.arange(0, 1440, 10)
    W = 1920
    H = 1080
    world_grid = np.stack(np.meshgrid(xi, yi, indexing='ij')).reshape([2, -1])
    world_coord = dataset.get_worldcoord_from_worldgrid(world_grid)
    range_masks = []
    for cam in range(dataset.num_cam):
        img = np.array(Image.open(f'/home/jiahao/Downloads/data/Wildtrack/Image_subsets/C{cam+1}/{image_idx:08d}.png'))
        
        u, v = np.meshgrid(np.arange(0, W, 1), np.arange(0, H, 1))
        fx, cx, fy, cy = dataset.intrinsic_matrices[cam][0, 0], dataset.intrinsic_matrices[cam][0, 2], dataset.intrinsic_matrices[cam][1, 1], dataset.intrinsic_matrices[cam][1, 2]
        dx = (u - cx) / fx
        dy = (v - cy) / fy
        dir_cam = np.stack([dx, dy, np.ones_like(dx)], axis=2)
        dir_cam = dir_cam / np.linalg.norm(dir_cam, axis=2, keepdims=True)
        dir_cam = dir_cam.reshape(-1, 3)
        w2c = np.eye(4)
        w2c[:3] = dataset.extrinsic_matrices[cam]
        c2w = np.linalg.inv(w2c)
        T = c2w[:3, 3]
        R = c2w[:3, :3]
        dir_world = R @ dir_cam.T
        t = - T[2] / dir_world[2]
        X = T[0] + t * dir_world[0]
        Y = T[1] + t * dir_world[1]
        
        ground_range_x = [world_coord[0].min(), world_coord[0].max()]
        ground_range_y = [world_coord[1].min(), world_coord[1].max()]
        range_mask = (X > ground_range_x[0]) & (X < ground_range_x[1]) & (Y > ground_range_y[0]) & (Y < ground_range_y[1])
        
        range_mask = range_mask.reshape(H, W).astype(np.uint8)
        range_masks.append(range_mask)
    return range_masks


def ground_mask_vis():
    root = r'/home/jiahao/3DReconstruction/mvdet/wildtrack_data/masks/ground/00000000'
    img_root = r'/home/jiahao/3DReconstruction/mvdet/wildtrack_data/Image_subsets/00000000'
    range_mask = define_ground_range()
            
    for mask_path, image_path, gt_mask in zip(sorted(os.listdir(root)), sorted(os.listdir(img_root)), range_mask):
        mask_path = os.path.join(root, mask_path)
        image_path = os.path.join(img_root, image_path)

        predicted_range_mask = np.load(mask_path)
        gt_mask = gt_mask.astype(np.bool_)
        predicted_range_mask = predicted_range_mask.astype(np.bool_)
        
        combined_mask = gt_mask & predicted_range_mask
        
        image = Image.open(image_path)
        # add the combined mask to the image and the mask green
        image = np.array(image)
        image[combined_mask, 1] = 255
        # visualize the gt_mask, predicted_range_mask and combined_mask
        
        fig, ax = plt.subplots(1, 3, figsize=(15, 5))
        ax[0].imshow(gt_mask, cmap='gray')
        ax[0].set_title('Ground Truth Ground Mask')
        ax[0].axis('off')
        ax[1].imshow(predicted_range_mask, cmap='gray')
        ax[1].set_title('Predicted Ground Mask')
        ax[1].axis('off')
        ax[2].imshow(image)
        ax[2].set_title('Image with Combined Mask')
        ax[2].axis('off')
        plt.tight_layout()
        plt.savefig(f"multiview_detector/results/ground_mask/{os.path.basename(mask_path).replace('.npy', '.jpg')}", dpi=300, pad_inches=0, bbox_inches='tight')
        
        
def pedestrian_mask_vis(vis=False):
    people_root = r'/home/jiahao/3DReconstruction/mvdet/wildtrack_data/masks/people/00000000'
    ground_root = r'/home/jiahao/3DReconstruction/mvdet/wildtrack_data/masks/ground/00000000'
    img_root = r'/home/jiahao/3DReconstruction/mvdet/wildtrack_data/Image_subsets/00000000'
    masks = []
    images = []
    image_paths = []
    for ground_mask_path, people_mask_path, image_path in zip(sorted(os.listdir(ground_root)), sorted(os.listdir(people_root)), sorted(os.listdir(img_root))):
        ground_mask_path = os.path.join(ground_root, ground_mask_path)
        people_mask_path = os.path.join(people_root, people_mask_path)
        image_path = os.path.join(img_root, image_path)
        image = Image.open(image_path)
        image_paths.append(image_path)
        predicted_ground_range_mask = np.load(ground_mask_path)
        predicted_ground_range_mask = predicted_ground_range_mask.astype(np.bool_)
        
        predicted_people_range_mask = np.load(people_mask_path)
        predicted_people_range_mask = predicted_people_range_mask.astype(np.bool_)
        intersection_mask = predicted_ground_range_mask & predicted_people_range_mask
        predicted_people_range_mask = predicted_people_range_mask & ~intersection_mask
        masks.append(predicted_people_range_mask)
        # add the combined mask to the image and the mask green
        image = np.array(image)
        images.append(image.copy())
        image[predicted_people_range_mask, 1] = 255
        
        if vis:
            # visualize the gt_mask, combined_mask
            fig, ax = plt.subplots(1, 2, figsize=(10, 5))
            ax[0].imshow(predicted_people_range_mask, cmap='gray')
            ax[0].set_title('Predicted Pedestrian Mask')
            ax[0].axis('off')
            ax[1].imshow(image)
            ax[1].set_title('Image with Pedestrian Mask')
            ax[1].axis('off')
            plt.tight_layout()
            plt.show()
            # plt.savefig(f"multiview_detector/results/people_mask/{os.path.basename(ground_mask_path).replace('.npy', '.jpg')}", dpi=300, pad_inches=0, bbox_inches='tight')
    return masks, images, image_paths
if __name__ == '__main__':
    pedestrian_mask_vis(True)