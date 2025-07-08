import os, json, re, sys
from cv2 import norm
sys.path.append(os.getcwd())

import numpy as np
import cv2, torch
from PIL import Image
import xml.etree.ElementTree as ET
from torch.utils.data import Dataset
import copy
import cv2
import numpy as np
import random
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset

def flip_joints(joints, joints_vis, width, pairs):
    joints[:, 0] = width - joints[:, 0] - 1

    for pair in pairs:
        joints[pair[0], :], joints[pair[1], :] = \
            joints[pair[1], :], joints[pair[0], :].copy()
        joints_vis[pair[0], :], joints_vis[pair[1], :] = \
            joints_vis[pair[1], :], joints_vis[pair[0], :].copy()

    return joints, joints_vis

def get_3rd_point(a, b):
    direct = a - b
    return b + np.array([-direct[1], direct[0]], dtype=np.float32)


def get_dir(src_point, rot_rad):
    sn, cs = np.sin(rot_rad), np.cos(rot_rad)

    src_result = [0, 0]
    src_result[0] = src_point[0] * cs - src_point[1] * sn
    src_result[1] = src_point[0] * sn + src_point[1] * cs

    return src_result
    
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

class JointsDataset(Dataset):

    def __init__(self, DATASET, stage, transform=None):
        self.stage = stage 
        assert self.stage in ('train', 'val', 'test')

        self.transform = transform
        self.data = list()

        self.keypoint_num = DATASET.KEYPOINT.NUM
        self.flip_pairs = DATASET.KEYPOINT.FLIP_PAIRS
        self.upper_body_ids = DATASET.KEYPOINT.UPPER_BODY_IDS
        self.lower_body_ids = DATASET.KEYPOINT.LOWER_BODY_IDS
        self.kp_load_min_num = DATASET.KEYPOINT.LOAD_MIN_NUM

        self.input_shape = DATASET.INPUT_SHAPE
        self.output_shape = DATASET.OUTPUT_SHAPE
        self.w_h_ratio = DATASET.WIDTH_HEIGHT_RATIO 

        self.pixel_std = DATASET.PIXEL_STD
        self.color_rgb = DATASET.COLOR_RGB

        self.basic_ext = DATASET.TRAIN.BASIC_EXTENTION
        self.rand_ext = DATASET.TRAIN.RANDOM_EXTENTION
        self.x_ext = DATASET.TRAIN.X_EXTENTION
        self.y_ext = DATASET.TRAIN.Y_EXTENTION
        self.scale_factor_low = DATASET.TRAIN.SCALE_FACTOR_LOW
        self.scale_factor_high = DATASET.TRAIN.SCALE_FACTOR_HIGH
        self.scale_shrink_ratio = DATASET.TRAIN.SCALE_SHRINK_RATIO
        self.rotation_factor = DATASET.TRAIN.ROTATION_FACTOR
        self.prob_rotation = DATASET.TRAIN.PROB_ROTATION
        self.prob_flip = DATASET.TRAIN.PROB_FLIP
        self.num_keypoints_half_body = DATASET.TRAIN.NUM_KEYPOINTS_HALF_BODY
        self.prob_half_body = DATASET.TRAIN.PROB_HALF_BODY
        self.x_ext_half_body = DATASET.TRAIN.X_EXTENTION_HALF_BODY
        self.y_ext_half_body = DATASET.TRAIN.Y_EXTENTION_HALF_BODY
        self.add_more_aug = DATASET.TRAIN.ADD_MORE_AUG
        self.gaussian_kernels = DATASET.TRAIN.GAUSSIAN_KERNELS

        self.test_x_ext = DATASET.TEST.X_EXTENTION
        self.test_y_ext = DATASET.TEST.Y_EXTENTION

    def __len__(self):
        return self.data_num


    def __getitem__(self, idx):
        d = copy.deepcopy(self.data[idx])

        img_id = d['img_id']
        img_path = d['img_path']

        data_numpy = cv2.imread(img_path, cv2.IMREAD_COLOR)

        if data_numpy is None:
            raise ValueError('fail to read {}'.format(img_path))

        if self.color_rgb:
            data_numpy = cv2.cvtColor(data_numpy, cv2.COLOR_BGR2RGB)

        joints = d['joints'][:, :2]
        joints_vis = d['joints'][:, -1].reshape((-1, 1))
        
        center = d['center']
        scale = d['scale']
        score = d['score'] if 'score' in d else 1
        rotation = 0

        if self.stage == 'train':
            scale[0] *= (1 + self.basic_ext)
            scale[1] *= (1 + self.basic_ext)
            rand = np.random.rand() if self.rand_ext else 1.0
            scale[0] *= (1 + rand * self.x_ext)
            rand = np.random.rand() if self.rand_ext else 1.0
            scale[1] *= (1 + rand * self.y_ext)
        else:
            scale[0] *= (1 + self.test_x_ext)
            scale[1] *= (1 + self.test_y_ext)

        # fit the ratio
        if scale[0] > self.w_h_ratio * scale[1]:
            scale[1] = scale[0] * 1.0 / self.w_h_ratio
        else:
            scale[0] = scale[1] * 1.0 * self.w_h_ratio

        # augmentation
        if self.stage == 'train':
            # half body
            # if (np.sum(joints_vis[:, 0] > 0) > self.num_keypoints_half_body
            #     and np.random.rand() < self.prob_half_body):
            #     c_half_body, s_half_body = self.half_body_transform(
            #         joints, joints_vis)

            #     if c_half_body is not None and s_half_body is not None:
            #         center, scale = c_half_body, s_half_body

            # scale
            rand = random.uniform(
                    1 + self.scale_factor_low, 1 + self.scale_factor_high)
            scale_ratio = self.scale_shrink_ratio * rand
            scale *= scale_ratio

            rotation
            if random.random() <= self.prob_rotation:
                rotation = random.uniform(
                        -self.rotation_factor, self.rotation_factor)

            # flip
            # if random.random() <= self.prob_flip:
            #     data_numpy = data_numpy[:, ::-1, :]
            #     joints, joints_vis = flip_joints(
            #         joints, joints_vis, data_numpy.shape[1], self.flip_pairs)
            #     center[0] = data_numpy.shape[1] - center[0] - 1

        trans = get_affine_transform(center, scale, rotation, self.input_shape)

        img = cv2.warpAffine(
            data_numpy,
            trans,
            (int(self.input_shape[1]), int(self.input_shape[0])),
            flags=cv2.INTER_LINEAR)
        if self.transform:
            img = self.transform(img)

        if self.stage == 'train':
            for i in range(self.keypoint_num):
                if joints_vis[i, 0] > 0:
                    joints[i, 0:2] = affine_transform(joints[i, 0:2], trans)
                    if joints[i, 0] < 0 \
                            or joints[i, 0] > self.input_shape[1] - 1 \
                            or joints[i, 1] < 0 \
                            or joints[i, 1] > self.input_shape[0] - 1:
                        joints_vis[i, 0] = 0 # 0
            valid = torch.from_numpy(joints_vis).float()

            labels_num = len(self.gaussian_kernels)
            labels = np.zeros(
                    (labels_num, self.keypoint_num, *self.output_shape))
            for i in range(labels_num):
                labels[i] = self.generate_heatmap(
                        joints, valid, kernel=self.gaussian_kernels[i])
            labels = torch.from_numpy(labels).float()
            return img, valid, labels
        else:
            joints[0, 0:2] = affine_transform(joints[0, 0:2], trans) # only one keypoint by default
            return img, score, center, scale, img_id, joints, data_numpy

    def _get_data(self):
        raise NotImplementedError

    def evaluate(self):
        raise NotImplementedError

    def half_body_transform(self, joints, joints_vis):
        upper_joints = []
        lower_joints = []
        for joint_id in range(self.keypoint_num):
            if joints_vis[joint_id, 0] > 0:
                if joint_id in self.upper_body_ids:
                    upper_joints.append(joints[joint_id])
                else:
                    lower_joints.append(joints[joint_id])

        if np.random.randn() < 0.5 and len(upper_joints) > 3:
            selected_joints = upper_joints
        else:
            selected_joints = lower_joints \
                if len(lower_joints) > 3 else upper_joints

        if len(selected_joints) < 3:
            return None, None

        selected_joints = np.array(selected_joints, dtype=np.float32)

        left_top = np.amin(selected_joints, axis=0)
        right_bottom = np.amax(selected_joints, axis=0)

        center = (left_top + right_bottom) / 2

        w = right_bottom[0] - left_top[0]
        h = right_bottom[1] - left_top[1]

        rand = np.random.rand()
        w *= (1 + rand * self.x_ext_half_body)
        rand = np.random.rand()
        h *= (1 + rand * self.y_ext_half_body)

        if w > self.w_h_ratio * h:
            h = w * 1.0 / self.w_h_ratio
        elif w < self.w_h_ratio * h:
            w = h * self.w_h_ratio

        scale = np.array([w * 1.0 / self.pixel_std, h * 1.0 / self.pixel_std],
            dtype=np.float32)

        return center, scale
    
    def generate_heatmap(self, joints, valid, kernel=(7, 7)):
        heatmaps = np.zeros(
                (self.keypoint_num, *self.output_shape), dtype='float32')

        for i in range(self.keypoint_num):
            if valid[i] < 1:
                continue
            target_y = joints[i, 1] * self.output_shape[0] \
                    / self.input_shape[0]
            target_x = joints[i, 0] * self.output_shape[1] \
                    / self.input_shape[1]
            heatmaps[i, int(target_y), int(target_x)] = 1
            heatmaps[i] = cv2.GaussianBlur(heatmaps[i], kernel, 0)
            maxi = np.amax(heatmaps[i])
            if maxi <= 1e-8:
                continue
            heatmaps[i] /= maxi / 255

        return heatmaps 


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

def to_numpy(data):
    if isinstance(data, np.ndarray):
        return data
    elif isinstance(data, torch.Tensor):
        return data.detach().cpu().numpy()
    else:
        return np.array(data)


def floorGrid_to_floorDepthMap(pts, extrinsic, intrinsic, HW, dataset='Wildtrack'):
    pts = np.array(pts)
    pts_hom = np.concatenate([pts, np.ones((pts.shape[0], 1))], axis=1)
    cam_pts = np.dot(extrinsic, pts_hom.T)
    img_pts = np.dot(intrinsic, cam_pts)
    img_pts /= img_pts[2, :] # W H 1
    mask = (img_pts[0, :] >= 0) & (img_pts[0, :] <= HW[1]) & (img_pts[1, :] >= 0) & (img_pts[1, :] <= HW[0])
    if dataset == 'Wildtrack':
        mask = mask & (cam_pts[2, :] > 0)
    else:
        mask = mask
    img_pts, cam_pts = img_pts.T, cam_pts.T
    depth_map = np.concatenate([img_pts[mask, 0:2], cam_pts[mask, 2].reshape(-1, 1)], axis=1)
    return depth_map.T    


def dense_depth_map(Pts, H, W, grid=1, normalize='max_norm'):
    """
        Args:
            Pts: [x, y, depth] -> [W, H]
            H, W: the height and width of image
            grid: the window size
    """
    assert normalize in ['max_norm', 'max_min_norm', None], 'Unknown Normalization'
    ng = 2 * grid + 1

    mX = np.zeros(shape=(H, W)) + np.float64('inf')
    mY = np.zeros(shape=(H, W)) + np.float64('inf')
    mD = np.zeros(shape=(H, W))

    mX[np.int32(Pts[1]), np.int32(Pts[0])] = Pts[0] - np.int32(Pts[0])
    mY[np.int32(Pts[1]), np.int32(Pts[0])] = Pts[1] - np.int32(Pts[1])
    mD[np.int32(Pts[1]), np.int32(Pts[0])] = Pts[2]

    KmX = np.zeros(shape=(ng, ng, H - ng, W - ng))
    KmY = np.zeros(shape=(ng, ng, H - ng, W - ng))
    KmD = np.zeros(shape=(ng, ng, H - ng, W - ng))

    # store distance between each grid pixel and grid center
    for i in range(ng):
        for j in range(ng):
            KmX[i, j] = mX[i : (H - ng + i), j : (W - ng + j)] + i
            KmY[i, j] = mY[i : (H - ng + i), j : (W - ng + j)] + j
            KmD[i, j] = mD[i : (H - ng + i), j : (W - ng + j)]
    
    Y = np.zeros_like(KmY[0, 0]) # store weight
    X = np.zeros_like(KmX[0, 0]) # store weighted distance
    
    # inverse distance weighted
    for i in range(ng):
        for j in range(ng):
            s = 1 / np.sqrt(KmX[i, j]**2 + KmY[i, j]**2)
            Y += s
            X += s * KmD[i, j]

    depth_map = np.zeros((H, W))
    Y[Y == 0] = 1
    depth_map[grid : -grid - 1, grid : -grid - 1] = X / Y
    if normalize == 'max_norm':
        max_depth = np.max(depth_map)
        depth_map = depth_map / np.max(depth_map) * 255
        return depth_map, max_depth
    elif normalize == 'max_min_norm':
        max_depth = np.min(depth_map)
        min_depth = np.max(depth_map)
        depth_map = (depth_map - min_depth) / (max_depth - min_depth) * 255
        return depth_map, max_depth, min_depth
    return depth_map


def project(proj, pts):
    pts_hom = np.insert(pts, 3, 1)
    pts = proj @ pts_hom
    pts /= pts[-1]
    return pts[:2]

intrinsic_camera_matrix_filenames = ['intr_CVLab1.xml', 'intr_CVLab2.xml', 'intr_CVLab3.xml', 'intr_CVLab4.xml',
                                     'intr_IDIAP1.xml', 'intr_IDIAP2.xml', 'intr_IDIAP3.xml']
extrinsic_camera_matrix_filenames = ['extr_CVLab1.xml', 'extr_CVLab2.xml', 'extr_CVLab3.xml', 'extr_CVLab4.xml',
                                     'extr_IDIAP1.xml', 'extr_IDIAP2.xml', 'extr_IDIAP3.xml']

WILDTRACK_BBOX_LABEL_NAMES = ['Person']

class Wildtrack(JointsDataset):
    grid_reduce = 4
    img_reduce = 4
    norm = 'max_norm'
    def __init__(self, 
                       DATASET, 
                       stage='train', 
                       transform=None,  
                       world_size = [480, 1440],
                       img_size = [1080, 1920],
                       reload_GK=False,
                       load_outside=True):
        super().__init__(DATASET, stage, transform)
        # WILDTRACK has ij-indexing: H*W=480*1440, so x should be \in [0,480), y \in [0,1440)
        # WILDTRACK has in-consistent unit: centi-meter (cm) for calibration & pos annotation
        self.stage = stage
        self.load_outside = load_outside
        self.root = r'F:\ANU\ENGN8602\Data\Wildtrack'
        self.reload_GK = reload_GK
        self.__name__ = 'Wildtrack'
        self.norm = Wildtrack.norm
        self.num_cam, self.num_frame = 7, 2000
        self.img_size, self.world_size = img_size, world_size # H,W; N_row,N_col
        self.grid_reduce, self.img_reduce = Wildtrack.grid_reduce, Wildtrack.img_reduce 
        self.reduced_grid_size = list(map(lambda x: int(x / self.grid_reduce), self.world_size)) # 120, 360
        self.label_names = WILDTRACK_BBOX_LABEL_NAMES
        self.intrinsic_matrices, self.extrinsic_matrices = zip(
            *[self.get_intrinsic_extrinsic_matrix(cam) for cam in range(self.num_cam)])
    
        self.labels_bbox, self.labels_pos, self.img_paths = self.download()
        self.data = self._get_data()
        self.data_num = len(self.data)
        
    def _get_data(self):
        data = list()
        # 0 ~ 398: train; 398 ~ 400: val
        if self.stage == 'train':
            self.labels_bbox = self.labels_bbox[:-2] 
            self.labels_pos = self.labels_pos[:-2]
            self.img_paths = self.img_paths[:-2]
        elif self.stage == 'val':
            self.labels_bbox = self.labels_bbox[-2:] 
            self.labels_pos = self.labels_pos[-2:]
            self.img_paths = self.img_paths[-2:]

        id = 0
        for labels_bbox, img_paths in zip(self.labels_bbox, self.img_paths):
            for c in range(1, 8):
                for bboxes, img_path in zip(labels_bbox[c], img_paths[c]):
                    img_name = img_path
                    x1,y1,x2,y2 = bboxes[:4]
                    w = x2 - x1
                    h = y2 - y1
                    bbox = np.array([x1, y1, w, h])
                    area = 0.0
                    joints = np.ones((1, 3), dtype=np.float32) * 2.0 # default 2, 2 means visable
                    joints[:, :2] = bboxes[-2:]
                    headRect = np.array([0, 0, 1, 1], np.int32)

                    center, scale = self._bbox_to_center_and_scale(bbox)

                    d = dict(aid=id,
                         area=area,
                         bbox=bbox,
                         center=center,
                         headRect=headRect,
                         img_id=id,
                         img_name=img_name,
                         img_path=img_path,
                         joints=joints,
                         scale=scale)
                
                    data.append(d)
                id += 1
        return data

    def gen_data_per_frame(self):
        per_frame_infos = list()
        for cam, bboxes in self.labels_bbox[0].items(): # first frame C1 ~ C2
            bbox_infos = list()
            for bbox in bboxes:
                x1,y1,x2,y2, kyp_x, kyp_y = bbox
                w = x2 - x1
                h = y2 - y1
                bbox = np.array([x1, y1, w, h])
                joints = np.ones((3), dtype=np.float32) * 2.0 # default 2, 2 means visable
                joints[:2] = [kyp_x, kyp_y]
                center, scale = self._bbox_to_center_and_scale(bbox)
                bbox_infos.append(dict(bbox=bbox,
                                  center=center,
                                  scale=scale,
                                  joints=joints))
            per_frame_infos.append([bbox_infos, self.img_paths[0][cam][0]])
        return per_frame_infos
                    
    def _bbox_to_center_and_scale(self, bbox):
        x, y, w, h = bbox

        center = np.zeros(2, dtype=np.float32)
        center[0] = x + w / 2.0
        center[1] = y + h / 2.0

        scale = np.array([w * 1.0 / self.pixel_std, h * 1.0 / self.pixel_std],
                dtype=np.float32)

        return center, scale

    def get_image_fpaths(self, frame_range):
        img_fpaths = {cam: {} for cam in range(1, self.num_cam+1)}
        for camera_folder in sorted(os.listdir(os.path.join(self.root, 'Image_subsets'))):
            cam = int(camera_folder[-1]) 
            if cam >= self.num_cam+1:
                continue
            for fname in sorted(os.listdir(os.path.join(self.root, 'Image_subsets', camera_folder))):
                frame = int(fname.split('.')[0])
                if frame in frame_range:
                    img_fpaths[cam][frame] = os.path.join(self.root, 'Image_subsets', camera_folder, fname)
        return img_fpaths
    
    @staticmethod
    def get_worldgrid_from_pos(pos):
        grid_x = pos % 480
        grid_y = pos // 480
        return np.array([grid_x, grid_y], dtype=int)
    
    @staticmethod
    def get_pos_from_worldgrid(worldgrid):
        grid_x, grid_y = worldgrid
        return grid_x + grid_y * 480

    @staticmethod
    def get_worldgrid_from_worldcoord(world_coord):
        # datasets default unit: centimeter & origin: (-300,-900)
        coord_x, coord_y = world_coord
        grid_x = (coord_x + 300) / 2.5
        grid_y = (coord_y + 900) / 2.5
        return np.array([grid_x, grid_y], dtype=int)
    
    @staticmethod
    def get_worldgrid_from_worldcoord_Tensor(world_coord):
        # world_coord: [n, 3] x, y, z
        # datasets default unit: centimeter & origin: (-300,-900)
        world_coord[:, 0] = (world_coord[:, 0] + 300) / 2.5
        world_coord[:, 1] = (world_coord[:, 1] + 900) / 2.5
        world_coord[:, 2] = world_coord[:, 2] / 2.5
        return world_coord
    
    @staticmethod
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
                grid_x, grid_y = self.get_worldgrid_from_pos(single_pedestrian['positionID'])
                og_gt.append(np.array([frame, grid_x, grid_y]))
        og_gt = np.stack(og_gt, axis=0)
        os.makedirs(os.path.dirname(self.gt_fpath), exist_ok=True)
        np.savetxt(self.gt_fpath, og_gt, '%d')

    def download(self):

        labels_bbox = list()
        labels_pos = list()
        labels_path = list()
        # if GK not exist (true), build GK; else, load GK from file. (GK: gaussian kernel heatmap)
        # BuildGK = self.reload_GK or not self.GK.GKExist() 
        for fname in sorted(os.listdir(os.path.join(self.root, 'annotations_positions'))):
            with open(os.path.join(self.root, 'annotations_positions', fname)) as json_file:
                all_pedestrians = json.load(json_file)

            each_cam_infos_for_bbox = {i: np.zeros((0, 6)) for i in range(1, self.num_cam+1)}
            each_cam_infos_for_pos = {i: np.zeros((0, 3)) for i in range(1, self.num_cam+1)}
            each_cam_infos_for_img_path = {i: list() for i in range(1, self.num_cam+1)}
            for single_pedestrian in all_pedestrians:
                x, y = self.get_worldgrid_from_pos(single_pedestrian['positionID'])
                for view in single_pedestrian['views']:
                    if view['xmax'] == view['ymax'] == view['xmin'] == view['ymin'] == -1:
                        continue
                    # bbox format: x1, y1, x2, y2, offset (offset between bbox center x and feet x). 
                    feet_coord = self.get_worldcoord_from_pos(single_pedestrian['positionID'])
                    feet_coord = np.insert(feet_coord, 2, 0)
                    feet_pts = project(proj = self.intrinsic_matrices[view['viewNum']] @ self.extrinsic_matrices[view['viewNum']],
                                       pts = feet_coord)
              
                    each_cam_infos_for_bbox[view['viewNum']+1] = np.append(each_cam_infos_for_bbox[view['viewNum']+1], 
                                                             np.stack([[np.float32(view['xmin']), np.float32(view['ymin']),
                                                             np.float32(view['xmax']), np.float32(view['ymax']), feet_pts[0], feet_pts[1]]]), axis=0)
                    each_cam_infos_for_pos[view['viewNum']+1] = np.append(each_cam_infos_for_pos[view['viewNum']+1],
                                                             np.array([[x, y, 0]]), axis=0)
                    each_cam_infos_for_img_path[view['viewNum']+1] += [os.path.join(self.root, 'Image_subsets', 'C'+str(view['viewNum']+1), fname.split('.')[0]+'.png')]
 
            if self.load_outside:
                try:
                    with open(os.path.join(self.root, 'annotations_positions_outside', fname[:-5] + "_outside.json")) as json_file:
                        all_pedestrians_outside = json.load(json_file)
                    for percam_pedestrian_outside in all_pedestrians_outside:
                        cam_id = percam_pedestrian_outside['cam_id']
                        for view_id, each_pedestrain in enumerate(percam_pedestrian_outside['outsiders']):
                            if each_pedestrain['xmax'] == each_pedestrain['ymax'] == each_pedestrain['xmin'] == each_pedestrain['ymin'] == -1:
                                continue
                            # append bbox and standing point on 2D image 
                            each_cam_infos_for_bbox[cam_id] = np.append(each_cam_infos_for_bbox[cam_id], 
                                    np.array([[each_pedestrain['xmin'], each_pedestrain['ymin'], each_pedestrain['xmax'], each_pedestrain['ymax'], *each_pedestrain['standing_point_2d']]]), axis=0)
                            each_cam_infos_for_img_path[cam_id] += [os.path.join(root, 'Image_subsets', 'C'+str(cam_id), fname.split('.')[0]+'.png')]
                except:
                    pass

            labels_bbox.append(each_cam_infos_for_bbox)
            labels_pos.append(each_cam_infos_for_pos)
            labels_path.append(each_cam_infos_for_img_path)
        
        return labels_bbox, labels_pos, labels_path
    
    @staticmethod
    def get_offset( cent_x, feet_x, w):
        return ( cent_x - feet_x ) / w
    
    @staticmethod
    def reverse_offset(cent_x, offset, w):
        return ( cent_x - offset * w )

    def pts2depth(self, pos, extrinsic):
        worldcoord = np.zeros((4), dtype=np.float64)
        worldcoord[:2] = self.get_worldcoord_from_worldgrid(pos)
        worldcoord[-1] = 1
        cam_pts = np.dot(extrinsic, worldcoord.reshape(-1, 1))
        return cam_pts[2]

    def get_global_cam_coord_range(self, grids):
        # calculate the range of global camera coordinate systems
        x_min, x_max = min(grids[0, :]), max(grids[0, :])
        y_min, y_max = min(grids[1, :]), max(grids[1, :])
        z_min, z_max = min(grids[2, :]), max(grids[2, :])
        return np.array([x_min, x_max, y_min, y_max, z_min, z_max])
    
    def get_floor_depth_map(self):
        depth_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'depth_map')
        if not os.path.exists(depth_root):
            os.mkdir(depth_root)
        depth_map_save_root = os.path.join(depth_root, self.__name__)
        if not os.path.exists(depth_map_save_root):
            os.mkdir(depth_map_save_root)
        world_floor_grid = to_numpy(make_grid(world_size=self.world_size, cube_LW=[1,1], dataset='Wildtrack')).reshape(-1, 3)
        floor_depth_map_list, depth_max_list, grid_range_list = list(), list(), list()
        for cam in range(0, self.num_cam):
            if not os.path.exists(os.path.join(depth_map_save_root, 'c%d_depth_map_floor.npz'%cam)):
                # world_pts_grid = np.zeros_like(world_floor_grid)
                world_pts_grid = Wildtrack.get_worldcoord_from_worldgrid(to_numpy(world_floor_grid.T))
                grid_range = self.get_global_cam_coord_range(world_pts_grid)
                floor_depth_map = floorGrid_to_floorDepthMap(world_pts_grid.T, self.extrinsic_matrices[cam], 
                                                        self.intrinsic_matrices[cam], self.img_size)
                
                floor_depth_map, depth_max = dense_depth_map(floor_depth_map, self.img_size[0], self.img_size[1], 5) # grid = 5!
                save_path = os.path.join(depth_map_save_root, 'c%d_depth_map_floor.npz'%cam)
                np.savez(save_path, depth_map_floor=floor_depth_map, depth_max=depth_max, grid_range=grid_range)
                print('Cam {}\'s depth infos saved in {}'.format(cam, save_path))
            else:
                data = np.load(os.path.join(depth_map_save_root, 'c%d_depth_map_floor.npz'%cam))
                floor_depth_map = data['depth_map_floor']
                depth_max = data['depth_max']
                grid_range = data['grid_range']
            floor_depth_map_list.append(floor_depth_map)
            depth_max_list.append(depth_max)
            grid_range_list.append(grid_range)
        floor_depth_map_list = [torch.from_numpy(dp) for dp in floor_depth_map_list]
        depth_max_list = [torch.tensor(dm) for dm in depth_max_list]
        grid_range_list = [torch.from_numpy(gr) for gr in grid_range_list]

        return floor_depth_map_list, depth_max_list, grid_range_list



    