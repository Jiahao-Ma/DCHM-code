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

if __name__ == '__main__':
    image_idx = 0
    print('Predicting depth using depth-anything/Depth-Anything-V2-Large-hf...')
    depth_model = pipeline(task="depth-estimation", model="depth-anything/Depth-Anything-V2-Large-hf", device=0)
    metric_depth_model = get_model(encoder='vitl')
    depth_type = 'rel_depth'
    data_root = '/home/jiahao/Downloads/data/wildtrack_data'
    wildtrack_data = Wildtrack(data_root, mask_type='people', mask_label_type='split', start_with=0, end_with=1)
    xi = np.arange(0, 480, 10)
    yi = np.arange(0, 1440, 10)
    world_grid = np.stack(np.meshgrid(xi, yi, indexing='ij')).reshape([2, -1])
    world_coord = get_worldcoord_from_worldgrid(world_grid)
    
    scales = []
    shifts = []
    dis2depth = DisparityToDepth()
    data = wildtrack_data[image_idx]

    images = data['nonnorm_image']
    for cam in range(wildtrack_data.num_cam):
        if depth_type == 'rel_depth':
            imagePIL = Image.fromarray(images[cam].transpose(1, 2, 0).astype(np.uint8))
            w, h = imagePIL.size
            disp = depth_model(imagePIL)['predicted_depth']
            # disp = (disp - disp.min()) / (disp.max() - disp.min())
            disp = F.interpolate(disp[:, None], (h, w), mode="bilinear", align_corners=True)[0, 0]
            # pred_depth, mask = dis2depth(disp)
            pred_depth = 1 / disp
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
        world_coord_copy = world_coord[:, mask].copy()
        gt_depth = gt_depth[mask]
        img_coord = np.round(img_coord).astype(np.int32).transpose()
        image = cv2.cvtColor(images[cam].transpose(1, 2, 0).astype(np.uint8), cv2.COLOR_RGB2BGR)
        for point in img_coord:
            cv2.circle(image, tuple(point.astype(int)), 2, (0, 255, 0), -1)

        cv2.namedWindow('image')
        cv2.setMouseCallback('image', draw_rectangle)
        selected_points = []
        selected_gt_depths = []
        while True:
            img_copy = image.copy()
            if drawing:
                cv2.rectangle(img_copy, (ix, iy), (ex, ey), (0, 255, 0), 2)
                left_top = np.array([np.min([ix, ex]), np.min([iy, ey])])
                right_bottom = np.array([np.max([ix, ex]), np.max([iy, ey])])
                within_masks = np.logical_and( np.logical_and( img_coord[:, 0] > left_top[0], img_coord[:, 0] < right_bottom[0]), 
                                np.logical_and( img_coord[:, 1] > left_top[1], img_coord[:, 1] < right_bottom[1]))
                if len(selected_points) == 0:
                    selected_points = img_coord[within_masks]
                    selected_gt_depths = gt_depth[within_masks]
                else:
                    selected_points = np.concatenate([selected_points, img_coord[within_masks]], axis=0)
                    selected_gt_depths = np.concatenate([selected_gt_depths, gt_depth[within_masks]], axis=0)
            for p in selected_points:
                cv2.circle(img_copy, tuple(p), 2, (0, 0, 255), -1)
            cv2.imshow('image', img_copy)
            k = cv2.waitKey(1) & 0xFF
            if k == 27:
                break
        # calculate the scale
        
        selected_pred_depth = pred_depth[selected_points[:, 1], selected_points[:, 0]] #TODO: CHECK?
        # scale = selected_gt_depths / selected_pred_depth
        # scales.append(np.median(scale))
        # print(f'cam{cam+1} mean scale: {np.mean(scale):.4f}; median scale: {np.median(scale):.4f}; std scale: {np.std(scale):.4f}')
        
        # calculate shift and scale using least square
        A = np.vstack([selected_pred_depth, np.ones(len(selected_pred_depth))]).T
        scale, shift = np.linalg.lstsq(A, selected_gt_depths, rcond=None)[0]
        print(f'cam{cam+1} scale: {scale:.4f}; shift: {shift:.4f}')
        scales.append(scale)
        shifts.append(shift)
        
        # reset the drawing
        ix, iy = -1, -1
        ex, ey = -1, -1
        selected_points = []
        selected_gt_depths = []
            
        cv2.destroyAllWindows()
        pass
    #save the scales
    scales = np.array(scales)
    shifts = np.array(shifts)
    # np.save('depth_wt/scales.npy', scales)
    np.save('depth_wt/shifts_scales.npy', {'scales': scales, 'shifts': shifts})