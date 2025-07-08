import os,sys;sys.path.append(os.getcwd())
import numpy as np
import cv2
import re
import pickle
import math
import torch
import time
import json
import glob
from PIL import Image
import matplotlib.pyplot as plt
from scipy.sparse import coo_matrix
from torchvision.transforms import ToTensor
from scipy.stats import multivariate_normal

import xml.etree.ElementTree as ET
from torchvision.datasets import VisionDataset
from torchvision.transforms import Compose
from submodules.depth_anything_v2.util.transform import Resize, NormalizeImage, PrepareForNet, Crop

intrinsic_camera_matrix_filenames = ['intr_CVLab1.xml', 'intr_CVLab2.xml', 'intr_CVLab3.xml', 'intr_CVLab4.xml',
                                     'intr_IDIAP1.xml', 'intr_IDIAP2.xml', 'intr_IDIAP3.xml']
extrinsic_camera_matrix_filenames = ['extr_CVLab1.xml', 'extr_CVLab2.xml', 'extr_CVLab3.xml', 'extr_CVLab4.xml',
                                     'extr_IDIAP1.xml', 'extr_IDIAP2.xml', 'extr_IDIAP3.xml']

def focal2fov(focal, pixels):
    return 2*math.atan(pixels/(2*focal))


def getProjectionMatrix(znear, zfar, K, h, w):
    near_fx = znear / K[0, 0]
    near_fy = znear / K[1, 1]
    left = - (w - K[0, 2]) * near_fx
    right = K[0, 2] * near_fx
    bottom = (K[1, 2] - h) * near_fy
    top = K[1, 2] * near_fy

    P = torch.zeros(4, 4)
    z_sign = 1.0
    P[0, 0] = 2.0 * znear / (right - left)
    P[1, 1] = 2.0 * znear / (top - bottom)
    P[0, 2] = (right + left) / (right - left)
    P[1, 2] = (top + bottom) / (top - bottom)
    P[3, 2] = z_sign
    P[2, 2] = z_sign * zfar / (zfar - znear)
    P[2, 3] = -(zfar * znear) / (zfar - znear)
    return P

def getWorld2View2(R, t, translate=np.array([.0, .0, .0]), scale=1.0):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = R.transpose()
    Rt[:3, 3] = t
    Rt[3, 3] = 1.0

    C2W = np.linalg.inv(Rt)
    cam_center = C2W[:3, 3]
    cam_center = (cam_center + translate) * scale
    C2W[:3, 3] = cam_center
    Rt = np.linalg.inv(C2W)
    return np.float32(Rt)

def is_08d_format(filename):
    try:
        return f'{int(filename):08d}' == filename
    except ValueError:
        return False


class Wildtrack(VisionDataset):
    def __init__(self, root, mask_type='ground', mode='train', mask_label_type='comb', render_engine='gsplat', start_with=0, end_with=None):
        super(Wildtrack, self).__init__(root)
        assert mask_label_type in ['split', 'comb'], 'mask_label_type should be split or comb'
        assert render_engine in ['gsplat', 'vallina'], 'render_engine should be gsplat or vallina gaussian'
        self.root = root
        self.mode = mode
        self.__name__ = 'Wildtrack'
        self.mask_type = mask_type
        self.render_engine = render_engine
        self.img_shape, self.worldgrid_shape = [1080, 1920], [480, 1440]  # H,W; N_row,N_col
        self.num_cam, self.num_frame = 7, 2000
        self.start_with = start_with # the start frame index
        self.end_with = end_with # the last frame index
        
        # initialize the camera information, image, mask paths 
        self.intrinsic_matrices, self.w2cs = zip(
            *[self.get_intrinsic_extrinsic_matrix(cam) for cam in range(self.num_cam)])
        
        self.get_image_fpaths()
        self.get_frame_idx()
        self.get_mask_fpaths(self.mask_type)
        self.mask_label_type = mask_label_type
        
        # The size is for depth map prediction
        # self.depth_map_size = [512, 512]
        self.depth_map_size = [540, 960] 
        self.mean=np.array([0.485, 0.456, 0.406]).reshape(3, 1, 1)
        self.std =np.array([0.229, 0.224, 0.225]).reshape(3, 1, 1)
        self.transform = Compose([
            Resize(
                width=self.depth_map_size[1],
                height=self.depth_map_size[0],
                resize_target=True if mode == 'train' else False,
                keep_aspect_ratio=True,
                # ensure_multiple_of=14,
                resize_method='lower_bound',
                image_interpolation_method=cv2.INTER_CUBIC,
            ),
            NormalizeImage(mean=self.mean.squeeze(), std=self.std.squeeze()),
            PrepareForNet(),
        ])
        
        # The size is for image feature extraction
        # self.image_feat_size = [512, 960]
        self.image_feat_size = self.depth_map_size # [540, 960]
        self.znear = 0.05 
        self.zfar = 10000.0
        self.trans = [0.0, 0.0, 0.0]
        self.scale = 1.0
        scale_x = self.image_feat_size[1] / self.img_shape[1]
        scale_y = self.image_feat_size[0] / self.img_shape[0]
        self.intrinsic_matrices = self.adjust_intrinsic_matrix(self.intrinsic_matrices, scale_x, scale_y)
        
        self.novel_view = []   
        for cam_idx, w2c in enumerate(self.w2cs):
            if self.render_engine == 'vallina':
            
                '''
                    Version1.0 (Fixed the bug. Ours)
                '''
                # FovX = focal2fov(self.intrinsic_matrices[cam_idx][0, 0], self.image_feat_size[1])
                # FovY = focal2fov(self.intrinsic_matrices[cam_idx][1, 1], self.image_feat_size[0])
                # c2w = np.linalg.inv(w2c).astype(np.float32)
                # intr = self.intrinsic_matrices[cam_idx]
                # projection_matrix = getProjectionMatrix(znear=self.znear, zfar=self.zfar, K=intr, h=self.image_feat_size[0], w=self.image_feat_size[1]).transpose(0, 1)
                '''
                    # Fixed: Find the problem here. `world_view_transform` = w2c.T
                '''
                # world_view_transform = torch.tensor(w2c).transpose(0, 1) 
                # full_proj_transform = (world_view_transform.unsqueeze(0).bmm(projection_matrix.unsqueeze(0))).squeeze(0)
                # camera_center = c2w[:3, 3]
                '''
                    Version2.0 (Fixed the bug in the above version)
                        Official GPSGaussian code
                '''
                height, width = self.image_feat_size    
                R = np.array(w2c[:3, :3], np.float32).reshape(3, 3).transpose(1, 0)
                T = np.array(w2c[:3, 3], np.float32)
                intr = self.intrinsic_matrices[cam_idx]
                FovX = focal2fov(intr[0, 0], width)
                FovY = focal2fov(intr[1, 1], height)
                projection_matrix = getProjectionMatrix(znear=self.znear, zfar=self.zfar, K=intr, h=height, w=width).transpose(0, 1)
                world_view_transform = torch.tensor(getWorld2View2(R, T, np.array(self.trans), self.scale)).transpose(0, 1)
                full_proj_transform = (world_view_transform.unsqueeze(0).bmm(projection_matrix.unsqueeze(0))).squeeze(0)
                camera_center = world_view_transform.inverse()[3, :3]
                c2w = np.linalg.inv(w2c).astype(np.float32)
                self.novel_view.append({
                    'view_id': torch.IntTensor([cam_idx]),
                    'extr': torch.FloatTensor(w2c),
                    'intr': torch.FloatTensor(intr),
                    'FovX': FovX,
                    'FovY': FovY,
                    'width': self.image_feat_size[1],
                    'height': self.image_feat_size[0],
                    'world_view_transform': world_view_transform, # w2c
                    'full_proj_transform': full_proj_transform,
                    'camera_center': camera_center,
                    'c2w': torch.FloatTensor(c2w)
                })
            elif self.render_engine == 'gsplat':
                self.novel_view.append({
                    'width': self.image_feat_size[1],
                    'height': self.image_feat_size[0],
                    'w2c': torch.FloatTensor(w2c),
                    'c2w': torch.FloatTensor(np.linalg.inv(w2c).astype(np.float32)),
                    'K': torch.FloatTensor(self.intrinsic_matrices[cam_idx]),
                })
        camera_locations = []
        for cam_idx in range(self.num_cam):
            camera_locations.append(self.novel_view[cam_idx]['c2w'][:3, 3])
        camera_locations = np.stack(camera_locations, axis=0)
        scene_center = np.mean(camera_locations, axis=0).reshape(-1, 3)    
        dist = np.linalg.norm(camera_locations - scene_center, axis=1)
        self.scene_scale = np.max(dist)
           
    def get_intrinsic_extrinsic_matrix(self, camera_i):
        intrinsic_camera_path = os.path.join(self.root, 'calibrations', 'intrinsic_zero')
        intrinsic_params_file = cv2.FileStorage(os.path.join(intrinsic_camera_path,
                                                             intrinsic_camera_matrix_filenames[camera_i]),
                                                flags=cv2.FILE_STORAGE_READ)
        intrinsic_matrix = intrinsic_params_file.getNode('camera_matrix').mat()
        intrinsic_params_file.release()

        extrinsic_params_file_root = ET.parse(os.path.join(self.root, 'calibrations', 'extrinsic',
                                                           extrinsic_camera_matrix_filenames[camera_i])).getroot()

        rvec = extrinsic_params_file_root.findall('rvec')[0].text.lstrip().rstrip().split(' ')
        rvec = np.array(list(map(lambda x: float(x), rvec)), dtype=np.float32)

        tvec = extrinsic_params_file_root.findall('tvec')[0].text.lstrip().rstrip().split(' ')
        tvec = np.array(list(map(lambda x: float(x), tvec)), dtype=np.float32)

        rotation_matrix, _ = cv2.Rodrigues(rvec)
        translation_matrix = np.array(tvec, dtype=np.float32).reshape(3, 1)
        extrinsic_matrix = np.hstack((rotation_matrix, translation_matrix))
        extrinsic_matrix = np.vstack((extrinsic_matrix, np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32).reshape(1, 4)))
        return intrinsic_matrix, extrinsic_matrix
    
    def get_image_fpaths(self):
        self.image_paths = []
        
        for image_folder_name in sorted(os.listdir(os.path.join(self.root, 'Image_subsets'))):
            if not is_08d_format(image_folder_name): 
                continue
            image_folder_path = os.path.join(self.root, 'Image_subsets', image_folder_name)
            per_scene_image_paths = []
            for image_name in sorted(os.listdir(image_folder_path)):
                image_path = os.path.join(image_folder_path, image_name)
                per_scene_image_paths.append(image_path)
            self.image_paths.append(per_scene_image_paths)
            
        if self.end_with is None or self.end_with == -1:
            self.image_paths = self.image_paths[self.start_with:]
        else:
            self.image_paths = self.image_paths[self.start_with:self.end_with]
    
    def get_frame_idx(self):
        self.frame_idxs = []
        for image_folder_name in sorted(os.listdir(os.path.join(self.root, 'Image_subsets'))):
            if not is_08d_format(image_folder_name): 
                continue
            self.frame_idxs.append(image_folder_name)
        if self.end_with is None or self.end_with == -1:
            self.frame_idxs = self.frame_idxs[self.start_with:]
        else:
            self.frame_idxs = self.frame_idxs[self.start_with:self.end_with]
        
    def get_mask_fpaths(self, mask_type='ground'):
    
        self.mask_paths = []
        assert os.path.exists(os.path.join(self.root, 'masks', mask_type)), 'masks folder does not exist'
        for mask_folder_name in sorted(os.listdir(os.path.join(self.root, 'masks', mask_type))):
            if not is_08d_format(mask_folder_name): 
                continue
            mask_folder_path = os.path.join(self.root, 'masks', mask_type, mask_folder_name)
            per_scene_mask_paths = []
            for mask_path in sorted(glob.glob(os.path.join(mask_folder_path, '*.npy'))):
                if 'mask' in os.path.basename(mask_path):
                    continue
                per_scene_mask_paths.append(mask_path)
            self.mask_paths.append(per_scene_mask_paths)
        
        if self.end_with is None or self.end_with == -1:
            self.mask_paths = self.mask_paths[self.start_with:]
        else:
            self.mask_paths = self.mask_paths[self.start_with:self.end_with]
        
    def cal_ground_depth(self, HW=[1080, 1920]):
        def _get_worldcoord_from_worldgrid(worldgrid):
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
        xi = np.arange(0, 480, 0.1)
        yi = np.arange(0, 1440, 0.1)
        world_grid = np.stack(np.meshgrid(xi, yi, indexing='ij')).reshape([2, -1])
        zi = np.zeros_like(world_grid[0])
        world_grid = np.concatenate([world_grid, zi.reshape(1, -1)], axis=0)
        world_coord = _get_worldcoord_from_worldgrid(world_grid)
        H, W = HW
        
        depths = []
        for cam in range(self.num_cam):
            w2c = self.w2cs[cam]
            K = self.intrinsic_matrices[cam]
            pts_cam = w2c[:3, :3] @ world_coord + w2c[:3, 3].reshape(3, 1)
            pts_depth = pts_cam[-1]
            pts_scn = K @ pts_cam
            pts_scn = pts_scn[:2] / pts_scn[2]
            mask = (pts_scn[0] > 0) & (pts_scn[0] < W) & (pts_scn[1] > 0) & (pts_scn[1] < H) & (pts_depth > 0)
            pts_scn = pts_scn[:, mask]
            # render depth
            depth = pts_depth[mask]
            depth_canvas = np.full((H, W), np.inf)
            x_coords = pts_scn[0].astype(int)
            y_coords = pts_scn[1].astype(int)
            np.minimum.at(depth_canvas, (y_coords, x_coords), depth)
            
            depth_canvas = np.where(depth_canvas == np.inf, 0, depth_canvas)
            depths.append(depth_canvas)
        self.background_depths = np.stack(depths, axis=0)
        background_depth_path = os.path.join(self.root, 'depths', 'background', 'depths.npy')
        np.save(background_depth_path, self.background_depths)
        print('Background depth has been saved to {}'.format(background_depth_path))
    
    def adjust_intrinsic_matrix(self, intrinsic_matrix, scale_x, scale_y):
        for intrin_mat in intrinsic_matrix:
            intrin_mat[0, 0] *= scale_x
            intrin_mat[1, 1] *= scale_y
            intrin_mat[0, 2] *= scale_x
            intrin_mat[1, 2] *= scale_y
        return intrinsic_matrix
    
    def __len__(self) -> int:
        return len(self.image_paths)
    
    def __getitem__(self, index: int):
        samples = []
        for image_path, mask_path in zip(self.image_paths[index], self.mask_paths[index]):
            image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB) / 255.0
            if mask_path is None:
                mask = np.ones((self.img_shape[0], self.img_shape[1]), dtype=np.float32)
            else:
                if self.mask_label_type == 'split':
                    mask = np.load(mask_path)
                else:
                    raise NotImplementedError('mask_label_type should be split or comb')
            
            sample = self.transform({'image': image, 'mask':mask})
            sample['nonnorm_image'] = np.clip(sample['image'] * self.std + self.mean, 0, 1).astype(np.float32) * 255.0
            samples.append(sample)
        frame_idx = os.path.dirname(self.image_paths[index][0]).split('/')[-1]
        masks = []
        images = []
        nonnorm_images = []
        for sample in samples:
            images.append(sample['image'])
            masks.append(sample['mask'])
            nonnorm_images.append(sample['nonnorm_image'])
        try:
            masks = np.stack(masks, axis=0)
        except:
            pass
        return {'frame_idx': frame_idx,
                'image': np.stack(images, axis=0), 
                'nonnorm_image': np.stack(nonnorm_images, axis=0),
                'mask': masks,
                'novel_view': self.novel_view,
                'H': self.image_feat_size[0],
                'W': self.image_feat_size[1]
                }
        
class WildtrackDetection(VisionDataset):
    def __init__(self, root):
        super().__init__(root)
        # WILDTRACK has ij-indexing: H*W=480*1440, so x should be \in [0,480), y \in [0,1440)
        # WILDTRACK has in-consistent unit: centi-meter (cm) for calibration & pos annotation,
        self.__name__ = 'Wildtrack'
        self.img_shape, self.worldgrid_shape = [1080, 1920], [480, 1440]  # H,W; N_row,N_col
        self.num_cam, self.num_frame = 7, 2005
        # x,y actually means i,j in Wildtrack, which correspond to h,w
        self.indexing = 'ij'
        # i,j for world map indexing
        self.worldgrid2worldcoord_mat = np.array([[2.5, 0, -300], [0, 2.5, -900], [0, 0, 1]])
        self.intrinsic_matrices, self.extrinsic_matrices = zip(
            *[self.get_intrinsic_extrinsic_matrix(cam) for cam in range(self.num_cam)])

    def get_image_fpaths(self, frame_range):
        img_fpaths = {cam: {} for cam in range(self.num_cam)}
        for camera_folder in sorted(os.listdir(os.path.join(self.root, 'Image_subsets'))):
            cam = int(camera_folder[-1]) - 1
            if cam >= self.num_cam:
                continue
            for fname in sorted(os.listdir(os.path.join(self.root, 'Image_subsets', camera_folder))):
                frame = int(fname.split('.')[0])
                if frame in frame_range:
                    img_fpaths[cam][frame] = os.path.join(self.root, 'Image_subsets', camera_folder, fname)
        return img_fpaths

    def get_worldgrid_from_pos(self, pos):
        grid_x = pos % 480
        grid_y = pos // 480
        return np.array([grid_x, grid_y], dtype=int)

    def get_pos_from_worldgrid(self, worldgrid):
        grid_x, grid_y = worldgrid
        return grid_x + grid_y * 480

    def get_worldgrid_from_worldcoord(self, world_coord):
        # datasets default unit: centimeter & origin: (-300,-900)
        assert world_coord.shape[0] == 2 or world_coord.shape[0] == 3, 'world_coord should be 2D or 3D'
        if world_coord.shape[0] == 2:
            coord_x, coord_y = world_coord
            grid_x = (coord_x + 300) / 2.5
            grid_y = (coord_y + 900) / 2.5
            return np.array([grid_x, grid_y], dtype=int)
        else:
            coord_x, coord_y, coord_z = world_coord
            grid_x = (coord_x + 300) / 2.5
            grid_y = (coord_y + 900) / 2.5
            grid_z = coord_z / 2.5
            return np.array([grid_x, grid_y, grid_z], dtype=int)

    def get_worldcoord_from_worldgrid(self, worldgrid):
        # datasets default unit: centimeter & origin: (-300,-900)
        grid_x, grid_y = worldgrid
        coord_x = -300 + 2.5 * grid_x
        coord_y = -900 + 2.5 * grid_y
        return np.array([coord_x, coord_y])

    def get_worldcoord_from_pos(self, pos):
        grid = self.get_worldgrid_from_pos(pos)
        return self.get_worldcoord_from_worldgrid(grid)

    def get_pos_from_worldcoord(self, world_coord):
        grid = self.get_worldgrid_from_worldcoord(world_coord)
        return self.get_pos_from_worldgrid(grid)

    def get_intrinsic_extrinsic_matrix(self, camera_i):
        intrinsic_camera_path = os.path.join(self.root, 'calibrations', 'intrinsic_zero')
        intrinsic_params_file = cv2.FileStorage(os.path.join(intrinsic_camera_path,
                                                             intrinsic_camera_matrix_filenames[camera_i]),
                                                flags=cv2.FILE_STORAGE_READ)
        intrinsic_matrix = intrinsic_params_file.getNode('camera_matrix').mat()
        intrinsic_params_file.release()

        extrinsic_params_file_root = ET.parse(os.path.join(self.root, 'calibrations', 'extrinsic',
                                                           extrinsic_camera_matrix_filenames[camera_i])).getroot()

        rvec = extrinsic_params_file_root.findall('rvec')[0].text.lstrip().rstrip().split(' ')
        rvec = np.array(list(map(lambda x: float(x), rvec)), dtype=np.float32)

        tvec = extrinsic_params_file_root.findall('tvec')[0].text.lstrip().rstrip().split(' ')
        tvec = np.array(list(map(lambda x: float(x), tvec)), dtype=np.float32)

        rotation_matrix, _ = cv2.Rodrigues(rvec)
        translation_matrix = np.array(tvec, dtype=np.float32).reshape(3, 1)
        extrinsic_matrix = np.hstack((rotation_matrix, translation_matrix))

        return intrinsic_matrix, extrinsic_matrix

    def read_pom(self):
        bbox_by_pos_cam = {}
        cam_pos_pattern = re.compile(r'(\d+) (\d+)')
        cam_pos_bbox_pattern = re.compile(r'(\d+) (\d+) ([-\d]+) ([-\d]+) (\d+) (\d+)')
        with open(os.path.join(self.root, 'rectangles.pom'), 'r') as fp:
            for line in fp:
                if 'RECTANGLE' in line:
                    cam, pos = map(int, cam_pos_pattern.search(line).groups())
                    if pos not in bbox_by_pos_cam:
                        bbox_by_pos_cam[pos] = {}
                    if 'notvisible' in line:
                        bbox_by_pos_cam[pos][cam] = None
                    else:
                        cam, pos, left, top, right, bottom = map(int, cam_pos_bbox_pattern.search(line).groups())
                        bbox_by_pos_cam[pos][cam] = [max(left, 0), max(top, 0),
                                                     min(right, 1920 - 1), min(bottom, 1080 - 1)]
        return bbox_by_pos_cam

class frameDataset(VisionDataset):
    def __init__(self, base, train=True, train_and_val=True, transform=ToTensor(), target_transform=ToTensor(),
                 reID=False, grid_reduce=4, img_reduce=4, train_ratio=0.9, force_download=True):
        super().__init__(base.root, transform=transform, target_transform=target_transform)

        map_sigma, map_kernel_size = 20 / grid_reduce, 20
        img_sigma, img_kernel_size = 10 / img_reduce, 10
        self.reID, self.grid_reduce, self.img_reduce = reID, grid_reduce, img_reduce

        self.base = base
        self.root, self.num_cam, self.num_frame = base.root, base.num_cam, base.num_frame
        self.img_shape, self.worldgrid_shape = base.img_shape, base.worldgrid_shape  # H,W; N_row,N_col
        self.reducedgrid_shape = list(map(lambda x: int(x / self.grid_reduce), self.worldgrid_shape))

        if train_and_val:
            frame_range = range(0, self.num_frame)
        else:
            if train:
                frame_range = range(0, int(self.num_frame * train_ratio))
            else:
                frame_range = range(int(self.num_frame * train_ratio), self.num_frame)

        self.img_fpaths = self.base.get_image_fpaths(frame_range)
        self.map_gt = {}
        self.imgs_head_foot_gt = {}
        self.download(frame_range)

        self.gt_fpath = os.path.join(self.root, 'gt.txt')
        if not os.path.exists(self.gt_fpath) or force_download:
            self.prepare_gt()

        x, y = np.meshgrid(np.arange(-map_kernel_size, map_kernel_size + 1),
                           np.arange(-map_kernel_size, map_kernel_size + 1))
        pos = np.stack([x, y], axis=2)
        map_kernel = multivariate_normal.pdf(pos, [0, 0], np.identity(2) * map_sigma)
        map_kernel = map_kernel / map_kernel.max()
        kernel_size = map_kernel.shape[0]
        self.map_kernel = torch.zeros([1, 1, kernel_size, kernel_size], requires_grad=False)
        self.map_kernel[0, 0] = torch.from_numpy(map_kernel)

        x, y = np.meshgrid(np.arange(-img_kernel_size, img_kernel_size + 1),
                           np.arange(-img_kernel_size, img_kernel_size + 1))
        pos = np.stack([x, y], axis=2)
        img_kernel = multivariate_normal.pdf(pos, [0, 0], np.identity(2) * img_sigma)
        img_kernel = img_kernel / img_kernel.max()
        kernel_size = img_kernel.shape[0]
        self.img_kernel = torch.zeros([2, 2, kernel_size, kernel_size], requires_grad=False)
        self.img_kernel[0, 0] = torch.from_numpy(img_kernel)
        self.img_kernel[1, 1] = torch.from_numpy(img_kernel)
        pass

    def prepare_gt(self):
        og_gt = []
        for fname in sorted(os.listdir(os.path.join(self.root, 'annotations_positions'))):
            frame = int(fname.split('.')[0])
            with open(os.path.join(self.root, 'annotations_positions', fname)) as json_file:
                all_pedestrians = json.load(json_file)
            for single_pedestrian in all_pedestrians:
                def is_in_cam(cam):
                    return not (single_pedestrian['views'][cam]['xmin'] == -1 and
                                single_pedestrian['views'][cam]['xmax'] == -1 and
                                single_pedestrian['views'][cam]['ymin'] == -1 and
                                single_pedestrian['views'][cam]['ymax'] == -1)

                in_cam_range = sum(is_in_cam(cam) for cam in range(self.num_cam))
                if not in_cam_range:
                    continue
                grid_x, grid_y = self.base.get_worldgrid_from_pos(single_pedestrian['positionID'])
                og_gt.append(np.array([frame, grid_x, grid_y]))
        og_gt = np.stack(og_gt, axis=0)
        os.makedirs(os.path.dirname(self.gt_fpath), exist_ok=True)
        np.savetxt(self.gt_fpath, og_gt, '%d')

    def download(self, frame_range):
        for fname in sorted(os.listdir(os.path.join(self.root, 'annotations_positions'))):
            frame = int(fname.split('.')[0])
            if frame in frame_range:
                with open(os.path.join(self.root, 'annotations_positions', fname)) as json_file:
                    all_pedestrians = json.load(json_file)
                i_s, j_s, v_s = [], [], []
                head_row_cam_s, head_col_cam_s = [[] for _ in range(self.num_cam)], \
                                                 [[] for _ in range(self.num_cam)]
                foot_row_cam_s, foot_col_cam_s, v_cam_s = [[] for _ in range(self.num_cam)], \
                                                          [[] for _ in range(self.num_cam)], \
                                                          [[] for _ in range(self.num_cam)]
                for single_pedestrian in all_pedestrians:
                    x, y = self.base.get_worldgrid_from_pos(single_pedestrian['positionID'])
                    if self.base.indexing == 'xy':
                        i_s.append(int(y / self.grid_reduce))
                        j_s.append(int(x / self.grid_reduce))
                    else:
                        i_s.append(int(x / self.grid_reduce))
                        j_s.append(int(y / self.grid_reduce))
                    v_s.append(single_pedestrian['personID'] + 1 if self.reID else 1)
                    for cam in range(self.num_cam):
                        x = max(min(int((single_pedestrian['views'][cam]['xmin'] +
                                         single_pedestrian['views'][cam]['xmax']) / 2), self.img_shape[1] - 1), 0)
                        y_head = max(single_pedestrian['views'][cam]['ymin'], 0)
                        y_foot = min(single_pedestrian['views'][cam]['ymax'], self.img_shape[0] - 1)
                        if x > 0 and y > 0:
                            head_row_cam_s[cam].append(y_head)
                            head_col_cam_s[cam].append(x)
                            foot_row_cam_s[cam].append(y_foot)
                            foot_col_cam_s[cam].append(x)
                            v_cam_s[cam].append(single_pedestrian['personID'] + 1 if self.reID else 1)
                occupancy_map = coo_matrix((v_s, (i_s, j_s)), shape=self.reducedgrid_shape)
                self.map_gt[frame] = occupancy_map
                self.imgs_head_foot_gt[frame] = {}
                for cam in range(self.num_cam):
                    img_gt_head = coo_matrix((v_cam_s[cam], (head_row_cam_s[cam], head_col_cam_s[cam])),
                                             shape=self.img_shape)
                    img_gt_foot = coo_matrix((v_cam_s[cam], (foot_row_cam_s[cam], foot_col_cam_s[cam])),
                                             shape=self.img_shape)
                    self.imgs_head_foot_gt[frame][cam] = [img_gt_head, img_gt_foot]

    def __getitem__(self, index):
        frame = list(self.map_gt.keys())[index]
        imgs = []
        fpaths = []
        for cam in range(self.num_cam):
            fpath = self.img_fpaths[cam][frame]
            fpaths.append(fpath)
            img = Image.open(fpath).convert('RGB')
            if self.transform is not None:
                img = self.transform(img)
            imgs.append(img)
        imgs = torch.stack(imgs)
        map_gt = self.map_gt[frame].toarray()
        if self.reID:
            map_gt = (map_gt > 0).int()
        if self.target_transform is not None:
            map_gt = self.target_transform(map_gt)
        imgs_gt = []
        for cam in range(self.num_cam):
            img_gt_head = self.imgs_head_foot_gt[frame][cam][0].toarray()
            img_gt_foot = self.imgs_head_foot_gt[frame][cam][1].toarray()
            img_gt = np.stack([img_gt_head, img_gt_foot], axis=2)
            if self.reID:
                img_gt = (img_gt > 0).int()
            if self.target_transform is not None:
                img_gt = self.target_transform(img_gt)
            imgs_gt.append(img_gt.float())
        return imgs, map_gt.float(), imgs_gt, frame, fpaths 

    def __len__(self):
        return len(self.map_gt.keys())

def get_worldcoord_from_imagecoord(image_coord, intrinsic_mat, extrinsic_mat):
    project_mat = intrinsic_mat @ extrinsic_mat
    project_mat = np.linalg.inv(np.delete(project_mat, 2, 1))
    image_coord = np.concatenate([image_coord, np.ones([1, image_coord.shape[1]])], axis=0)
    world_coord = project_mat @ image_coord
    world_coord = world_coord[:2, :] / world_coord[2, :]
    return world_coord

def get_imagecoord_from_worldcoord(world_coord, intrinsic_mat, extrinsic_mat):
    project_mat = intrinsic_mat @ extrinsic_mat
    project_mat = np.delete(project_mat, 2, 1)
    world_coord = np.concatenate([world_coord, np.ones([1, world_coord.shape[1]])], axis=0)
    image_coord = project_mat @ world_coord
    image_coord = image_coord[:2, :] / image_coord[2, :]
    return image_coord


class WildtrackDepthEstimation(VisionDataset):
    def __init__(self, root, depth_mask_type='matching_label', mode='train', size=None, mask_choice="background+foreground", round_name='1_1', data_ratio = None):
        super(WildtrackDepthEstimation, self).__init__(root)
        assert depth_mask_type in ['matching_label', 'sam_label', 'pseudo_label'], 'depth_mask_type should be matching_label or sam_label'
        assert mask_choice in ['background', 'foreground', 'background+foreground'], 'mask_choice should be background, foreground or background+foreground'
        self.data_ratio = data_ratio    
        self.root = root
        self.mode = mode
        self.__name__ = 'Wildtrack'
        self.depth_mask_type = depth_mask_type
        self.round_name = round_name    
        self.mask_choice = mask_choice  
        self.img_shape, self.worldgrid_shape = [1080, 1920], [480, 1440]  # H,W; N_row,N_col
        self.num_cam, self.num_frame = 7, 2000
        self.intrinsic_matrices, self.w2cs = zip(
            *[self.get_intrinsic_extrinsic_matrix(cam) for cam in range(self.num_cam)])
        
        background_depth_path = os.path.join(self.root, 'depths', 'background', 'depths.npy')
        if not os.path.exists(background_depth_path):
            self.cal_ground_depth(self.img_shape)
        else:
            self.background_depths = np.load(background_depth_path)
        self.define_ground_range()
        self.get_image_fpaths()
        self.get_foreground_depth_and_mask_fpaths()
        self.get_background_mask_fpaths()
        self.split_data()
        
        if size is None:
            self.depth_map_size = [540, 960] 
        else:
            self.depth_map_size = size
        self.mean=np.array([0.485, 0.456, 0.406]).reshape(3, 1, 1)
        self.std =np.array([0.229, 0.224, 0.225]).reshape(3, 1, 1)
        self.transform = Compose([
            Resize(
                width=self.depth_map_size[1],
                height=self.depth_map_size[0],
                resize_target=False,# if self.mode == 'train' else False,
                keep_aspect_ratio=True,
                # ensure_multiple_of=14,
                resize_method='lower_bound',
                image_interpolation_method=cv2.INTER_CUBIC,
            ),
            NormalizeImage(mean=self.mean.squeeze(), std=self.std.squeeze()),
            PrepareForNet(),
        ] + ([])) #+ ([Crop(size[0])] if self.mode == 'train' else []))
        # no crop no target resize
        
        self.image_feat_size = self.depth_map_size 
        scale_x = self.image_feat_size[1] / self.img_shape[1]
        scale_y = self.image_feat_size[0] / self.img_shape[0]
        self.intrinsic_matrices = self.adjust_intrinsic_matrix(self.intrinsic_matrices, scale_x, scale_y)
        self.novel_view = []
        for cam_idx, w2c in enumerate(self.w2cs):
            self.novel_view.append({
                'width': self.image_feat_size[1],
                'height': self.image_feat_size[0],
                'w2c': torch.FloatTensor(w2c),
                'c2w': torch.FloatTensor(np.linalg.inv(w2c).astype(np.float32)),
                'K': torch.FloatTensor(self.intrinsic_matrices[cam_idx]),
            })
        
    def split_data(self):
        if self.data_ratio is not None:
            data_len = len(self.image_paths)
            train_data_len = int(data_len * self.data_ratio)
            if self.mode == 'train':
                self.image_paths = self.image_paths[:train_data_len]
                self.foreground_depth_paths = self.foreground_depth_paths[:train_data_len]
                self.foreground_depth_mask_paths = self.foreground_depth_mask_paths[:train_data_len]
                self.foreground_split_depth_mask_paths = self.foreground_split_depth_mask_paths[:train_data_len]
                self.background_depth_mask_paths = self.background_depth_mask_paths[:train_data_len]
            elif self.mode == 'val':
                self.image_paths = self.image_paths[train_data_len:]
                self.foreground_depth_paths = self.foreground_depth_paths[train_data_len:]
                self.foreground_depth_mask_paths = self.foreground_depth_mask_paths[train_data_len:]
                self.foreground_split_depth_mask_paths = self.foreground_split_depth_mask_paths[train_data_len:]
                self.background_depth_mask_paths = self.background_depth_mask_paths[train_data_len:]
                
    def get_image_fpaths(self):
        self.image_paths = []
        for image_folder_name in sorted(os.listdir(os.path.join(self.root, 'Image_subsets'))):
            image_folder_path = os.path.join(self.root, 'Image_subsets', image_folder_name)
            per_scene_image_paths = []
            for image_name in sorted(os.listdir(image_folder_path)):
                image_path = os.path.join(image_folder_path, image_name)
                per_scene_image_paths.append(image_path)
                
            self.image_paths.append(per_scene_image_paths)
 
    def get_foreground_depth_and_mask_fpaths(self):
        self.foreground_depth_paths = []            
        self.foreground_depth_mask_paths = []
        self.foreground_split_depth_mask_paths = []
        if self.depth_mask_type == 'matching_label':
            for folder_name in sorted(os.listdir(os.path.join(self.root, 'depths'))):
                if folder_name == 'background':
                    continue
                depth_path = os.path.join(self.root, 'depths', folder_name, f'depth_{self.round_name}.npy')
                depth_mask_path = os.path.join(self.root, 'depths', folder_name, f'mask_{self.round_name}.npy')
                folder_path = os.path.join(self.root, 'masks', 'people', folder_name)
                split_depth_mask_path = [os.path.join(folder_path, f'cam{i+1}.npy') for i in range(self.num_cam)]
                self.foreground_depth_paths.append(depth_path)
                self.foreground_depth_mask_paths.append(depth_mask_path)
                self.foreground_split_depth_mask_paths.append(split_depth_mask_path)
                
        elif self.depth_mask_type == 'sam_label':
            for folder_name in sorted(os.listdir(os.path.join(self.root, 'depths'))):
                if folder_name == 'background':
                    continue
                depth_path = os.path.join(self.root, 'depths', folder_name, f'depth_{self.round_name}.npy')
                folder_path = os.path.join(self.root, 'masks', 'people', folder_name)
                depth_mask_path = [os.path.join(folder_path, f'cam{i+1}.npy') for i in range(self.num_cam)]
                split_depth_mask_path = [os.path.join(folder_path, f'cam{i+1}.npy') for i in range(self.num_cam)]
                
                self.foreground_depth_paths.append(depth_path)
                self.foreground_depth_mask_paths.append(depth_mask_path)
                self.foreground_split_depth_mask_paths.append(split_depth_mask_path)
        
        elif self.depth_mask_type == 'pseudo_label':
            for folder_name in sorted(os.listdir(os.path.join(self.root, 'depths'))):
                if folder_name == 'background':
                    continue 
                depth_path = os.path.join(self.root, 'depths', folder_name, f'depths_{self.round_name}.npy')
                folder_path = os.path.join(self.root, 'masks', 'people', folder_name) # the same as depth
                depth_mask_path = [os.path.join(folder_path, f'cam{i+1}.npy') for i in range(self.num_cam)]
                self.foreground_depth_paths.append(depth_path)
                self.foreground_depth_mask_paths.append(depth_mask_path)
                self.foreground_split_depth_mask_paths.append(None)
    def get_background_mask_fpaths(self):
        self.background_depth_mask_paths = []
        for folder_name in sorted(os.listdir(os.path.join(self.root, 'depths'))):
            if folder_name == 'background':
                    continue
            folder_path = os.path.join(self.root, 'masks', 'ground', folder_name)
            split_depth_mask_path = [os.path.join(folder_path, f'cam{i+1}.npy') for i in range(self.num_cam)]
            self.background_depth_mask_paths.append(split_depth_mask_path)  
        
    def get_intrinsic_extrinsic_matrix(self, camera_i):
        intrinsic_camera_path = os.path.join(self.root, 'calibrations', 'intrinsic_zero')
        intrinsic_params_file = cv2.FileStorage(os.path.join(intrinsic_camera_path,
                                                             intrinsic_camera_matrix_filenames[camera_i]),
                                                flags=cv2.FILE_STORAGE_READ)
        intrinsic_matrix = intrinsic_params_file.getNode('camera_matrix').mat()
        intrinsic_params_file.release()

        extrinsic_params_file_root = ET.parse(os.path.join(self.root, 'calibrations', 'extrinsic',
                                                           extrinsic_camera_matrix_filenames[camera_i])).getroot()

        rvec = extrinsic_params_file_root.findall('rvec')[0].text.lstrip().rstrip().split(' ')
        rvec = np.array(list(map(lambda x: float(x), rvec)), dtype=np.float32)

        tvec = extrinsic_params_file_root.findall('tvec')[0].text.lstrip().rstrip().split(' ')
        tvec = np.array(list(map(lambda x: float(x), tvec)), dtype=np.float32)

        rotation_matrix, _ = cv2.Rodrigues(rvec)
        translation_matrix = np.array(tvec, dtype=np.float32).reshape(3, 1)
        extrinsic_matrix = np.hstack((rotation_matrix, translation_matrix))
        extrinsic_matrix = np.vstack((extrinsic_matrix, np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32).reshape(1, 4)))
        return intrinsic_matrix, extrinsic_matrix
            
    def adjust_intrinsic_matrix(self, intrinsic_matrix, scale_x, scale_y):
        for intrin_mat in intrinsic_matrix:
            intrin_mat[0, 0] *= scale_x
            intrin_mat[1, 1] *= scale_y
            intrin_mat[0, 2] *= scale_x
            intrin_mat[1, 2] *= scale_y
        return intrinsic_matrix      

    def define_ground_range(self):
        assert self.background_depths is not None, 'Background depth has not been calculated'
        self.predefined_background_depth_masks = []
        for cam_idx in range(self.num_cam):
            depth = self.background_depths[cam_idx]
            depth_mask = np.zeros_like(depth, dtype=np.bool_)
            depth_mask[depth > 0] = True
            self.predefined_background_depth_masks.append(depth_mask)
            
    def cal_ground_depth(self, HW=[1080, 1920]):
        def _get_worldcoord_from_worldgrid(worldgrid):
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
        xi = np.arange(0, 480, 0.1)
        yi = np.arange(0, 1440, 0.1)
        world_grid = np.stack(np.meshgrid(xi, yi, indexing='ij')).reshape([2, -1])
        zi = np.zeros_like(world_grid[0])
        world_grid = np.concatenate([world_grid, zi.reshape(1, -1)], axis=0)
        world_coord = _get_worldcoord_from_worldgrid(world_grid)
        H, W = HW
        
        depths = []
        for cam in range(self.num_cam):
            w2c = self.w2cs[cam]
            K = self.intrinsic_matrices[cam]
            pts_cam = w2c[:3, :3] @ world_coord + w2c[:3, 3].reshape(3, 1)
            pts_depth = pts_cam[-1]
            pts_scn = K @ pts_cam
            pts_scn = pts_scn[:2] / pts_scn[2]
            mask = (pts_scn[0] > 0) & (pts_scn[0] < W) & (pts_scn[1] > 0) & (pts_scn[1] < H) & (pts_depth > 0)
            pts_scn = pts_scn[:, mask]
            # render depth
            depth = pts_depth[mask]
            depth_canvas = np.full((H, W), np.inf)
            x_coords = np.clip(np.round(pts_scn[0]).astype(int), 0, W-1)
            y_coords = np.clip(np.round(pts_scn[1]).astype(int), 0, H-1)
            np.minimum.at(depth_canvas, (y_coords, x_coords), depth)
            
            depth_canvas = np.where(depth_canvas == np.inf, 0, depth_canvas)
            depths.append(depth_canvas)
        self.background_depths = np.stack(depths, axis=0)
        background_depth_path = os.path.join(self.root, 'depths', 'background', 'depths.npy')
        if not os.path.exists(os.path.dirname(background_depth_path)):
            os.makedirs(os.path.dirname(background_depth_path), exist_ok=True)
        np.save(background_depth_path, self.background_depths)
        print('Background depth has been saved to {}'.format(background_depth_path))

    
    def __len__(self) -> int:
        return len(self.image_paths) 
    
    def __getitem__(self, index: int):
        # TODO: adjust the mask of background 
        samples = []
        # Load foreground depth and mask
        foreground_depth_mask_path, foreground_depth_path = self.foreground_depth_mask_paths[index], self.foreground_depth_paths[index]
        foreground_split_depth_mask_path = self.foreground_split_depth_mask_paths[index]
        if foreground_split_depth_mask_path is not None:
            foreground_split_depth_mask = [np.load(mask_path) for mask_path in foreground_split_depth_mask_path]
        else:
            foreground_split_depth_mask = None
        foreground_depths = np.load(foreground_depth_path)
        if self.depth_mask_type == 'matching_label':
            foreground_depth_masks = np.load(foreground_depth_mask_path)
        elif self.depth_mask_type == 'sam_label':
            foreground_depth_masks = [np.load(mask_path) for mask_path in foreground_depth_mask_path]
            foreground_depth_masks = [np.where(mask > 0, np.ones_like(mask), np.zeros_like(mask)) for mask in foreground_depth_masks]
            foreground_depth_masks = np.stack(foreground_depth_masks, axis=0)
        elif self.depth_mask_type == 'pseudo_label':
            foreground_depth_masks = [np.load(mask_path) for mask_path in foreground_depth_mask_path]
            foreground_depth_masks = [np.where(mask > 0, np.ones_like(mask), np.zeros_like(mask)) for mask in foreground_depth_masks]
            foreground_depth_masks = np.stack(foreground_depth_masks, axis=0)
            
        height, width = self.image_feat_size
        for idx, (image_path, background_depth_mask_path) in enumerate(zip(self.image_paths[index],  self.background_depth_mask_paths[index])):
            # Load image
            image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB) / 255.0
            ori_size = image.shape[:2]
            image = cv2.resize(image, (width, height), interpolation=cv2.INTER_LINEAR)

            # Load background depth mask and depth
            background_depth_mask = np.load(background_depth_mask_path)
            background_depth_mask = np.where(background_depth_mask == 1, np.ones_like(background_depth_mask), np.zeros_like(background_depth_mask)) 
            background_depth_mask = cv2.resize(background_depth_mask, (width, height), interpolation=cv2.INTER_NEAREST).astype(np.bool_)
  
            pdf_background_depth_mask = self.predefined_background_depth_masks[idx].astype(np.float32)
            pdf_background_depth_mask = cv2.resize(pdf_background_depth_mask, (width, height), interpolation=cv2.INTER_NEAREST).astype(np.bool_)
            background_depth_mask = background_depth_mask & pdf_background_depth_mask
            
            background_depth = cv2.resize(self.background_depths[idx], (width, height), interpolation=cv2.INTER_NEAREST)
            background_depth = np.where(background_depth_mask, background_depth, np.zeros_like(background_depth))
            
            # Load foreground depth mask and depth
            foreground_depth = foreground_depths[idx]
            foreground_depth_mask = foreground_depth_masks[idx]
            foreground_depth = cv2.resize(foreground_depth, (width, height), interpolation=cv2.INTER_NEAREST)
            foreground_depth_mask = cv2.resize(foreground_depth_mask.astype(np.float32), (width, height), interpolation=cv2.INTER_NEAREST).astype(np.bool_)
            foreground_depth = np.where(foreground_depth_mask, foreground_depth, np.zeros_like(foreground_depth))
            
            if self.mask_choice == 'background+foreground':
                # comb_depth = np.where(foreground_depth_mask, foreground_depth, background_depth)
                # comb_mask = np.where(foreground_depth_mask, foreground_depth_mask, background_depth_mask)
                comb_depth = np.where(background_depth_mask, background_depth, foreground_depth)
                comb_mask = np.where(background_depth_mask, background_depth_mask, foreground_depth_mask)
            elif self.mask_choice == 'foreground':
                comb_depth = foreground_depth
                comb_mask = foreground_depth_mask
            elif self.mask_choice == 'background':
                comb_depth = background_depth
                comb_mask = background_depth_mask

            comb_mask = np.where(comb_depth == 0., 0., comb_mask)
            comb_depth = np.where(comb_mask, comb_depth, 0.)
            bg_mask = background_depth_mask
            
            sample = self.transform({'image': image, 'depth':comb_depth, 'mask':comb_mask})
            if foreground_split_depth_mask is not None:          
                split_depth_mask = foreground_split_depth_mask[idx] 
                split_depth_mask = cv2.resize(split_depth_mask.astype(np.float32), (width, height), interpolation=cv2.INTER_NEAREST).astype(np.int32)
                sample['split_mask'] = split_depth_mask
            else: 
                sample['split_mask'] = None
            sample['bg_mask'] =bg_mask
            samples.append(sample)
        frame_idx = os.path.dirname(self.image_paths[index][0]).split('/')[-1]
        images = []
        depths = []
        masks = []
        split_mask = []
        bg_mask = []
        for sample in samples:
            images.append(sample['image'])
            depths.append(sample['depth'])
            masks.append(sample['mask'])
            if sample['split_mask'] is not None:
                split_mask.append(sample['split_mask'])
            else:
                split_mask.append(sample['mask'])
                
            bg_mask.append(sample['bg_mask'])
            
        images = np.stack(images, axis=0)
        depths = np.stack(depths, axis=0)  
        masks = np.stack(masks, axis=0)  
        split_mask = np.stack(split_mask, axis=0)

        return {'frame_idx': frame_idx,
                'image': images, 
                'depth': depths, 
                'valid_mask': masks, 
                'split_mask': split_mask,
                'bg_mask' : bg_mask,
                'H': self.image_feat_size[0],
                'W': self.image_feat_size[1],
                'novel_view': self.novel_view
                }
        
class SuperPixels(VisionDataset):
    def __init__(self, data_root, n_segments, HW, start_with=0, end_with=-1):
        super(SuperPixels, self).__init__(data_root)
        self.data_root = data_root
        self.folders = sorted(glob.glob(os.path.join(data_root, 'superpixels', '*', str(n_segments))))
        if end_with == -1:
            self.folders = self.folders[start_with:]
        else:
            self.folders = self.folders[start_with:end_with]
        self.new_hw = HW
    def __len__(self):
        return len(self.folders)
    
    def __getitem__(self, index):
        folder = self.folders[index]
        image_paths = sorted(glob.glob(os.path.join(folder, '*.png')))
        images = [cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)/255.0 for image_path in image_paths]
        ori_hw = images[0].shape[:2]
        # resize
        images = [cv2.resize(image, (self.new_hw[1], self.new_hw[0]), interpolation=cv2.INTER_LINEAR) for image in images]
        with open(os.path.join(folder, 'centroids.json'), 'r', encoding='utf-8') as f:
            data = json.load(f)
        centroids = [np.array(d) for d in data['centroids']] # v, u
        centroids_idx = [np.array(d).astype(np.int32) for d in data['centroids_idx']]
        
        for i in range(len(centroids)):
            if len(centroids[i]) == 0:
                continue
            centroids[i][0] = centroids[i][0] * self.new_hw[0] / ori_hw[0]
            centroids[i][1] = centroids[i][1] * self.new_hw[1] / ori_hw[1]
                
        return {'image': images, 'centroids': centroids, 'centroids_idx': centroids_idx}
    
if __name__ == '__main__':
    import open3d as o3d
    import matplotlib.pyplot as plt
    def depth_filter(depth, ratio=0.5, max_depth=30):
        if isinstance(depth, np.ndarray):
            depth = torch.tensor(depth)
        grad_x = torch.abs(depth[:, 1:] - depth[:, :-1])  # Horizontal gradient
        grad_y = torch.abs(depth[1:, :] - depth[:-1, :])  # Vertical gradient

        # Pad the gradients to maintain the same size as the input depth map
        grad_x = torch.nn.functional.pad(grad_x, (0, 1), mode='constant', value=0)  # Pad last column
        grad_y = torch.nn.functional.pad(grad_y, (0, 0, 0, 1), mode='constant', value=0)  # Pad last row

        # Combine the gradients (you can also use the maximum or another metric)
        grad_magnitude = grad_x + grad_y

        # Create the mask: True for regions with gradient < threshold, False otherwise
        if ratio is not None:
            threshold = grad_magnitude.mean() * ratio
            mask = grad_magnitude < threshold
        else:
            mask = torch.ones_like(grad_magnitude, dtype=torch.bool)
            
        if max_depth is not None:
            mask = mask & (depth < max_depth)

        return mask.cpu().numpy()
    
    trainset = WildtrackDepthEstimation("/home/jiahao/Downloads/data/wildtrack_data",
                    mode='val', size=[540, 960], depth_mask_type='matching_label')#depth_mask_type='sam_label')#size=[518, 518])
    sample = trainset[0]
    
    image = sample['image'].transpose(0, 2, 3, 1)
    image = (image * trainset.std.reshape(1, 1, 1, 3) + trainset.mean.reshape(1, 1, 1, 3))*255
    # image = sample['image'] * 255
    images = image.astype(np.uint8)
    depths = sample['depth']
    valid_masks = sample['valid_mask']
    # CHECK DEPTH
    # for i in range(7):
    #     plt.figure(figsize=(30, 10))
    #     plt.subplot(131)
    #     plt.imshow(images[i])
    #     plt.subplot(132)
    #     plt.imshow(depths[i])
    #     plt.subplot(133)
    #     plt.imshow(valid_masks[i])
    #     plt.show()
    
    # CHECK 3D Point
    all_pts_xyz = []
    all_pts_rgb = []
    for i in range(7):
        intrinsic_matrix = trainset.intrinsic_matrices[i]
        extrinsic_matrix = trainset.w2cs[i]
        v, u = np.where(valid_masks[i])
        mask = depth_filter(depths[i], ratio=0.6, max_depth=None)
        depth = np.where(mask, depths[i], 0)
        depth = depth[v, u]
        pts_rgb = images[i][v, u]
        fx, cx, fy, cy = intrinsic_matrix[0, 0], intrinsic_matrix[0, 2], intrinsic_matrix[1, 1], intrinsic_matrix[1, 2]
        dx = (u - cx) / fx
        dy = (v - cy) / fy
        dir_cam = np.stack([dx, dy, np.ones_like(dx)], axis=1)
        
        dir_cam = dir_cam.reshape(-1, 3)
        pts_cam = dir_cam * depth.reshape(-1, 1)
        
        # valid_depth_mask = depth != 0
        # valid_mask = valid_mask & valid_depth_mask
        w2c = extrinsic_matrix
        c2w = np.linalg.inv(w2c)
        pts_world = c2w[:3, :3] @ pts_cam.T + c2w[:3, 3:]
        all_pts_xyz.append(pts_world.T)
        all_pts_rgb.append(pts_rgb)
    all_pts_xyz = np.concatenate(all_pts_xyz, axis=0)
    all_pts_rgb = np.concatenate(all_pts_rgb, axis=0)
    
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(all_pts_xyz)
    pcd.colors = o3d.utility.Vector3dVector(all_pts_rgb / 255.0)
    o3d.visualization.draw_geometries([pcd])
    # downsampling
    # o3d.io.write_point_cloud("test.ply", pcd)
    