# find the ground plane grid points and calculate the scale
import os,sys;sys.path.append(os.getcwd())
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2
from dataset.wildtrack import Wildtrack
from ProteusLib import DisparityToDepth
from transformers import pipeline
import torch.nn.functional as F
from submodules.depth_anything_v2_metric.dpt import DepthAnythingV2
from dataset.wildtrack import WildtrackDepthEstimation

def get_model(encoder, dataset='vkitti', max_depth=80, base_dir=r'/home/jiahao/3DReconstruction/mvdet/Depth-Anything-V2/'):
    model_configs = {
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]}
    }
    model = DepthAnythingV2(**{**model_configs[encoder], 'max_depth': max_depth})
    model.load_state_dict(torch.load(os.path.join(base_dir, f'checkpoints/depth_anything_v2_metric_{dataset}_{encoder}.pth'), map_location='cpu'))
    model.eval()
    
    return model.cuda()

drawing = False # true if mouse is pressed
ix, iy = -1, -1
ex, ey = -1, -1

def draw_rectangle(event, x, y, flags, param):
    global ix, iy, drawing, ex, ey

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix, iy = x, y
        ex, ey = x, y  
    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing == True:
            ex, ey = x, y
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        ex, ey = x, y
        # print(f"Rectangle selected from ({ix}, {iy}) to ({ex}, {ey})")
        
def get_imagecoord_from_worldcoord3D(world_coord, intrinsic_mat, extrinsic_mat):
    if world_coord.shape[0] == 2:
        world_coord = np.concatenate([world_coord, np.zeros([1, world_coord.shape[1]])], axis=0)
    world_coord_hom = np.concatenate([world_coord, np.ones([1, world_coord.shape[1]])], axis=0).astype(np.float32)
    cam_coord = extrinsic_mat @ world_coord_hom
    depth = cam_coord[2, :]
    img_coord = intrinsic_mat @ cam_coord[:3]
    img_coord = img_coord[:2, :] / img_coord[2, :]
    return img_coord, depth, cam_coord

def get_worldcoord_from_worldgrid(worldgrid):
    # datasets default unit: centimeter & origin: (-300,-900)
    dim = worldgrid.shape[0]
    if dim == 2:
        grid_x, grid_y = worldgrid
        coord_x = -300 + 2.5 * grid_x
        coord_y = -900 + 2.5 * grid_y
        return np.array([coord_x, coord_y])
    elif dim == 3:
        grid_x, grid_y, grid_z = worldgrid
        coord_x = -300 + 2.5 * grid_x
        coord_y = -900 + 2.5 * grid_y
        coord_z = 2.5 * grid_z
        return np.array([coord_x, coord_y, coord_z])



def least_squares_fit(x, y):
    """
    Perform least squares fitting to find scale and shift.
    y = scale * x + shift
    """
    A = np.vstack([x, np.ones(len(x))]).T
    scale, shift = np.linalg.lstsq(A, y, rcond=None)[0]
    return scale, shift

if __name__ == '__main__':
    image_idx = 0
    print('Predicting depth using depth-anything/Depth-Anything-V2-Large-hf...')
    depth_model = pipeline(task="depth-estimation", model="depth-anything/Depth-Anything-V2-Large-hf", device=0)
    metric_depth_model = get_model(encoder='vitl')
    depth_type = 'rel_depth'
    data_root = '/home/jiahao/Downloads/data/wildtrack_data'
    wildtrack_data = Wildtrack(data_root, mask_type='ground', mask_label_type='split', start_with=0, end_with=1)
    
    xi = np.arange(0, 480, 1)
    yi = np.arange(0, 1440, 1)
    world_grid = np.stack(np.meshgrid(xi, yi, indexing='ij')).reshape([2, -1])
    world_coord = get_worldcoord_from_worldgrid(world_grid)
    
    all_scales = []
    all_shifts = []
    all_depths = []
    all_masks = []
    dis2depth = DisparityToDepth()
    data = wildtrack_data[image_idx]
    
    images = data['nonnorm_image']
    ground_mask = data['mask']
    for cam in range(wildtrack_data.num_cam):
        background_mask = ground_mask[cam]
        if depth_type == 'rel_depth':
            imagePIL = Image.fromarray(images[cam].transpose(1, 2, 0).astype(np.uint8))
            w, h = imagePIL.size
            disp = depth_model(imagePIL)['predicted_depth']
            disp = (disp - disp.min()) / (disp.max() - disp.min())
            disp = F.interpolate(disp[:, None], (h, w), mode="bilinear", align_corners=True)[0, 0]
            pred_depth, mask_depth = dis2depth(disp)
            # pred_depth = 1 / (disp+1e-6)
            # mask_depth = np.ones_like(pred_depth).astype(np.bool_)
            pred_depth = pred_depth.cpu().numpy()
        elif depth_type == 'metric_depth':
            imageNP = images[cam].transpose(1, 2, 0).astype(np.uint8)
            h, w = imageNP.shape[:2]
            pred_depth = metric_depth_model.infer_image(imageNP)
        img_coord, gt_depth, _ = get_imagecoord_from_worldcoord3D(world_coord, 
                                                                  wildtrack_data.novel_view[cam]['K'].numpy(), 
                                                                  wildtrack_data.novel_view[cam]['w2c'].numpy())
        mask = np.where((img_coord[0] > 0) & (img_coord[1] > 0) &
                                        (img_coord[0] < w-1) & (img_coord[1] < h-1))[0]
        img_coord = img_coord[:, mask]
        
        gt_depth = gt_depth[mask]
        img_coord = np.round(img_coord).astype(np.int32).transpose()
        mask_valid_ground_pc = background_mask[img_coord[:, 1], img_coord[:, 0]].astype(bool)
        img_coord = img_coord[mask_valid_ground_pc]
        gt_depth = gt_depth[mask_valid_ground_pc]
        
        v, u = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')
        vu = np.stack([u.flatten(), v.flatten()], axis=1)
        # devide the image into 16 x 16 patches and calculate the shift and scale for each patch
        patch_size = 128
        h_patches = h // patch_size
        w_patches = w // patch_size
        shifts = np.zeros((h // patch_size, w // patch_size))
        scales = np.zeros((h // patch_size, w // patch_size))
        for i in range(0, w_patches * patch_size, patch_size):
            for j in range(0, h_patches * patch_size, patch_size):
                # Get coordinates for the current patch
                patch_mask = (vu[:, 0] >= i) & (vu[:, 0] < i + patch_size) & \
                            (vu[:, 1] >= j) & (vu[:, 1] < j + patch_size)
                patch_coords = vu[patch_mask]
                
                # Find img_coord points that fall within this patch
                in_patch = np.all((img_coord >= [i, j]) & (img_coord < [i + patch_size, j + patch_size]), axis=1)
                patch_img_coord = img_coord[in_patch]
                patch_gt_depth = gt_depth[in_patch]
                
                if len(patch_img_coord) > 1:  # We need at least 2 points for least squares
                    # Get predicted depths for the points in this patch
                    patch_pred_depth = pred_depth[patch_img_coord[:, 1], patch_img_coord[:, 0]]
                    
                    # Calculate scale and shift using least squares
                    scale, shift = least_squares_fit(patch_pred_depth, patch_gt_depth)
                    error = np.abs(patch_gt_depth - (patch_pred_depth * scale + shift))
                    print('Patch:', i, j, 'Shift:', shift, 'Scale:', scale, 'Error:', np.mean(error))
                    # Store shift and scale
                    
                    shifts[j // patch_size, i // patch_size] = shift
                    scales[j // patch_size, i // patch_size] = scale
                else:
                    # If insufficient points in this patch, set default values
                    shifts[j // patch_size, i // patch_size] = 0
                    scales[j // patch_size, i // patch_size] = 1
                
        scales[scales == 1] = np.median(scales[scales != 1])
        shifts[shifts == 0] = np.median(shifts[shifts != 0])
        full_scales = np.zeros((h, w))
        full_shifts = np.zeros((h, w))
        for i in range(0, w_patches * patch_size, patch_size):
            for j in range(0, h_patches * patch_size, patch_size):
                # Get coordinates for the current patch
                patch_mask = (vu[:, 0] >= i) & (vu[:, 0] < i + patch_size) & \
                            (vu[:, 1] >= j) & (vu[:, 1] < j + patch_size)
                patch_coords = vu[patch_mask]
                full_scales[j:j+patch_size, i:i+patch_size] = scales[j // patch_size, i // patch_size]
                full_shifts[j:j+patch_size, i:i+patch_size] = shifts[j // patch_size, i // patch_size]
        all_scales.append(full_scales)
        all_shifts.append(full_shifts)
        all_depths.append(pred_depth)
        all_masks.append(mask_depth)
        # image = cv2.cvtColor(images[cam].transpose(1, 2, 0).astype(np.uint8), cv2.COLOR_RGB2BGR)
        # for point in img_coord:
        #     cv2.circle(image, tuple(point.astype(int)), 2, (0, 255, 0), -1)

        # cv2.namedWindow('image')
        # cv2.imshow('image', image)
        
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
    all_scales = np.stack(all_scales)
    all_shifts = np.stack(all_shifts)
    all_depths = np.stack(all_depths)
    all_masks = np.stack(all_masks)
    # save the scale and shift together with the image
    data = {'scales': all_scales, 'shifts': all_shifts, 'depths': all_depths, 'masks': all_masks}
    np.save(f'./depth_wt/{image_idx:08d}_scales_shifts.npy', data)
    
    