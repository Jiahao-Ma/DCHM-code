import os, json, re, sys

from cv2 import norm
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
sys.path.append(os.getcwd())
import cv2
import torch
import numpy as np
from PIL import Image
from torchvision.datasets import VisionDataset
from scipy.sparse import coo_matrix 

from lib.data.GK import GaussianKernel
from lib.utils.tool_utils import to_numpy, make_grid 
from model.ffe.pcl import floorGrid_to_floorDepthMap, dense_depth_map
from lib.utils.tool_utils import project

intrinsic_camera_matrix_filenames = ['intr_Camera1.xml', 'intr_Camera2.xml', 'intr_Camera3.xml', 'intr_Camera4.xml',
                                     'intr_Camera5.xml', 'intr_Camera6.xml']
extrinsic_camera_matrix_filenames = ['extr_Camera1.xml', 'extr_Camera2.xml', 'extr_Camera3.xml', 'extr_Camera4.xml',
                                     'extr_Camera5.xml', 'extr_Camera6.xml']

MULTIVIEWX_BBOX_LABEL_NAMES = ['Person']

class MultiviewX(VisionDataset):
    grid_reduce = 4
    img_reduce = 4
    norm = 'max_min_norm'
    def __init__(self, root, # The Path of MultiviewX
                       world_size = [640, 1000],
                       img_size = [1080, 1920],
                       force_download=False, 
                       reload_GK=False,
                       load_outside=False):
        super().__init__(root)
        # MultiviewX has xy-indexing: H*W=640*1000, thus x is \in [0,1000), y \in [0,640)
        # MultiviewX has consistent unit: meter (m) for calibration & pos annotation
        self.__name__ = 'MultiviewX'
        self.load_outside = load_outside
        self.norm = MultiviewX.norm
        self.num_cam, self.num_frame = 6, 400
        self.img_size, self.world_size = img_size, world_size # H,W; N_row,N_col
        self.grid_reduce, self.img_reduce = MultiviewX.grid_reduce, MultiviewX.img_reduce 
        self.reduced_grid_size = list(map(lambda x: int(x / self.grid_reduce), self.world_size)) # 160, 250
        self.reload_GK = reload_GK
        self.label_names = MULTIVIEWX_BBOX_LABEL_NAMES
        self.intrinsic_matrices, self.extrinsic_matrices = zip(
            *[self.get_intrinsic_extrinsic_matrix(cam) for cam in range(self.num_cam)])

        self.GK = GaussianKernel(save_dir=os.path.join(os.path.dirname(__file__), 'mx_GK.npz'))
        
        self.labels_bbox, self.labels_pos, batch_gt = self.download()
        self.heatmaps = batch_gt['heatmaps']

        # Create gt.txt file to evaluate MODA, MODP, prec, rcll metrics
        self.gt_fpath = os.path.join(self.root, 'gt.txt')
        if not os.path.exists(self.gt_fpath) or force_download:
            self.prepare_gt()

        # create floor depth map and get max depth value of each view
        self.depth_map, self.depth_max, self.depth_min, self.grid_range = self.get_floor_depth_map()


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
        grid_x = pos % 1000
        grid_y = pos // 1000
        return np.array([grid_x, grid_y], dtype=int)
    
    @staticmethod
    def get_pos_from_worldgrid(worldgrid):
        grid_x, grid_y = worldgrid
        return grid_x + grid_y * 1000

    @staticmethod
    def get_worldgrid_from_worldcoord(world_coord):
        coord_x, coord_y = world_coord
        grid_x = coord_x * 40
        grid_y = coord_y * 40
        return np.array([grid_x, grid_y], dtype=int)
    
    @staticmethod
    def get_worldgrid_from_worldcoord_Tensor(world_coord):
        world_coord[:, 0] = world_coord[:, 0] * 40
        world_coord[:, 1] = world_coord[:, 1] * 40
        world_coord[:, 2] = world_coord[:, 2] * 40
        return world_coord
    
    @staticmethod
    def get_worldcoord_from_worldgrid(worldgrid):
        dim = worldgrid.shape[0]
        if dim == 2:
            grid_x, grid_y = worldgrid
            coord_x = grid_x / 40
            coord_y = grid_y / 40
            return np.array([coord_x, coord_y])
        else:
            grid_x, grid_y, grid_z = worldgrid
            coord_x = grid_x / 40
            coord_y = grid_y / 40
            # coord_z = grid_z / 40
            coord_z = grid_z
            return np.array([coord_x, coord_y, coord_z])


    def get_worldcoord_from_pos(self, pos):
        grid = self.get_worldgrid_from_pos(pos)
        return self.get_worldcoord_from_worldgrid(grid)

    def get_pos_from_worldcoord(self, world_coord):
        grid = self.get_worldgrid_from_worldcoord(world_coord)
        return self.get_pos_from_worldgrid(grid)

    def get_intrinsic_extrinsic_matrix(self, camera_i):
        intrinsic_camera_path = os.path.join(self.root, 'calibrations', 'intrinsic')
        fp_calibration = cv2.FileStorage(os.path.join(intrinsic_camera_path,
                                                      intrinsic_camera_matrix_filenames[camera_i]),
                                         flags=cv2.FILE_STORAGE_READ)
        intrinsic_matrix = fp_calibration.getNode('camera_matrix').mat()
        fp_calibration.release()

        extrinsic_camera_path = os.path.join(self.root, 'calibrations', 'extrinsic')
        fp_calibration = cv2.FileStorage(os.path.join(extrinsic_camera_path,
                                                      extrinsic_camera_matrix_filenames[camera_i]),
                                         flags=cv2.FILE_STORAGE_READ)
        rvec, tvec = fp_calibration.getNode('rvec').mat().squeeze(), fp_calibration.getNode('tvec').mat().squeeze()
        fp_calibration.release()

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
                grid_x, grid_y = self.base.get_worldgrid_from_pos(single_pedestrian['positionID'])
                og_gt.append(np.array([frame, grid_x, grid_y]))
        og_gt = np.stack(og_gt, axis=0)
        os.makedirs(os.path.dirname(self.gt_fpath), exist_ok=True)
        np.savetxt(self.gt_fpath, og_gt, '%d')


    def download(self):

        labels_bbox = list()
        labels_pos = list()
        # if GK not exist (true), build GK; else, load GK from file. (GK: gaussian kernel heatmap)
        BuildGK = self.reload_GK or not self.GK.GKExist() 
        
        for fname in sorted(os.listdir(os.path.join(self.root, 'annotations_positions'))):
            # frame = int(fname.split('.')[0])
            with open(os.path.join(self.root, 'annotations_positions', fname)) as json_file:
                all_pedestrians = json.load(json_file)
            i_s, j_s, v_s, off_x, off_y = [], [], [], [], []
            each_cam_infos_for_bbox = {i: np.zeros((0, 6)) for i in range(1, self.num_cam+1)}
            each_cam_infos_for_pos = {i: np.zeros((0, 3)) for i in range(1, self.num_cam+1)}
            
            for single_pedestrian in all_pedestrians:
                x, y = self.get_worldgrid_from_pos(single_pedestrian['positionID'])
                for view in single_pedestrian['views']:
                    if view['xmax'] == view['ymax'] == view['xmin'] == view['ymin'] == -1:
                        continue
                    # bbox format: x1, y1, x2, y2, depth. 
                    feet_coord = self.get_worldcoord_from_pos(single_pedestrian['positionID'])
                    feet_coord = np.insert(feet_coord, 2, 0)
                    feet_pts = project(proj = self.intrinsic_matrices[view['viewNum']] @ self.extrinsic_matrices[view['viewNum']],
                                       pts = feet_coord)
                    
                    each_cam_infos_for_bbox[view['viewNum']+1] = np.append(each_cam_infos_for_bbox[view['viewNum']+1], 
                                                                np.array([[view['xmin'], view['ymin'],
                                                                view['xmax'], view['ymax'], feet_pts[0], feet_pts[1] ]]), axis=0)
                    
                    each_cam_infos_for_pos[view['viewNum']+1] = np.append(each_cam_infos_for_pos[view['viewNum']+1], 
                                                                np.array([[x, y, 0]]), axis=0)
                if BuildGK:
                    i_s.append(int(y / self.grid_reduce))
                    j_s.append(int(x / self.grid_reduce))
                    v_s.append(1)
                    off_x.append(x / self.grid_reduce - int(x / self.grid_reduce))
                    off_y.append(y / self.grid_reduce - int(y / self.grid_reduce))
                    
            if self.load_outside:     
                # load outside boxes
                if os.path.exists(os.path.join(self.root, 'annotations_positions_outside')):
                    with open(os.path.join(self.root, 'annotations_positions_outside', fname[:-5] + "_outside.json")) as json_file_outside:
                        all_pedestrians_outside = json.load(json_file_outside)
                    
                    for percam_pedestrian_outside in all_pedestrians_outside:
                        cam_id = percam_pedestrian_outside['cam_id']
                
                        for each_pedestrain in percam_pedestrian_outside['outsiders']:
                            if each_pedestrain['xmax'] == each_pedestrain['ymax'] == each_pedestrain['xmin'] == each_pedestrain['ymin'] == -1:
                                continue
                            each_cam_infos_for_bbox[cam_id]= np.append(each_cam_infos_for_bbox[cam_id], 
                                                                                    np.array([[each_pedestrain['xmin'], each_pedestrain['ymin'], each_pedestrain['xmax'], each_pedestrain['ymax'], *each_pedestrain['standing_point_2d']]]), axis=0)
                else:
                    print('Can\'t find outside annotation. Load data annotation error!')

                    
            if BuildGK:
                occupancy_map = coo_matrix((v_s, (i_s, j_s)), shape=self.reduced_grid_size).toarray()
                
                self.GK.add_item(occupancy_map)

            labels_bbox.append(each_cam_infos_for_bbox)
            labels_pos.append(each_cam_infos_for_pos)
            
        if BuildGK:
            # dump RGK to file
            batch_gt = self.GK.dump_to_file()
        else:
            batch_gt = self.GK.load_from_file()
        
        return labels_bbox, labels_pos, batch_gt
                
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

    # TODO: check xy
    def get_floor_depth_map(self):
        depth_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'depth_map')
        if not os.path.exists(depth_root):
            os.mkdir(depth_root)
        depth_map_save_root = os.path.join(depth_root, self.__name__)
        if not os.path.exists(depth_map_save_root):
            os.mkdir(depth_map_save_root)
        world_floor_grid = to_numpy(make_grid(world_size=self.world_size, cube_LW=[1,1], dataset='MultiviewX')).reshape(-1, 3)
        floor_depth_map_list, depth_max_list, depth_min_list, grid_range_list = list(), list(), list(), list()
        for cam in range(0, self.num_cam):
            if not os.path.exists(os.path.join(depth_map_save_root, 'c%d_depth_map_floor.npz'%cam)):
                world_pts_grid = np.zeros_like(world_floor_grid)
                world_pts_grid = MultiviewX.get_worldcoord_from_worldgrid(to_numpy(world_floor_grid.T))
                grid_range = self.get_global_cam_coord_range(world_pts_grid)
                floor_depth_map = floorGrid_to_floorDepthMap(world_pts_grid.T, self.extrinsic_matrices[cam], 
                                                        self.intrinsic_matrices[cam], self.img_size, 'MultiviewX')
                floor_depth_map, depth_max, depth_min = dense_depth_map(floor_depth_map, self.img_size[0], self.img_size[1], 5, 'max_min_norm') # grid = 5!
                np.savez(os.path.join(depth_map_save_root, 'c%d_depth_map_floor.npz'%cam), \
                        depth_map_floor=floor_depth_map, depth_max=depth_max, depth_min=depth_min, grid_range=grid_range)
            else:
                data = np.load(os.path.join(depth_map_save_root, 'c%d_depth_map_floor.npz'%cam))
                floor_depth_map = data['depth_map_floor']
                depth_max = data['depth_max']
                depth_min = data['depth_min']
                grid_range = data['grid_range']
            floor_depth_map_list.append(floor_depth_map)
            depth_max_list.append(depth_max)
            depth_min_list.append(depth_min)
            grid_range_list.append(grid_range)
        grid_range_list = [torch.from_numpy(gr) for gr in grid_range_list]
        floor_depth_map_list = [torch.from_numpy(dp) for dp in floor_depth_map_list]
        try:
            depth_max_list = [torch.Tensor(dm) for dm in depth_max_list]
            depth_min_list = [torch.Tensor(dm) for dm in depth_min_list]
        except:
            depth_max_list = [torch.tensor(dm) for dm in depth_max_list]
            depth_min_list = [torch.tensor(dm) for dm in depth_min_list]

        return floor_depth_map_list, depth_max_list, depth_min_list, grid_range_list

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from PIL import Image
    img_id = 0
    
    mc = MultiviewX(root='F:\ANU\ENGN8602\Data\MultiviewX')
    for cam in range(6):
        img_path = 'F:\\ANU\\ENGN8602\\Data\\MultiviewX\\Image_subsets\\C%d\\%04d.png'%(cam+1, img_id)
        img = Image.open(img_path)
        pos_list = mc.labels_pos[img_id][cam+1]
        depth_map = mc.depth_map[cam]
        depth_map = depth_map.detach().cpu().numpy()
        coord = mc.get_worldcoord_from_worldgrid(np.array(pos_list).T)
        coord = np.vstack([coord, np.ones((1, coord.shape[1]))])
        # proj_mat = mc.intrinsic_matrices[cam] @ mc.extrinsic_matrices[cam]
        img_coord = mc.extrinsic_matrices[cam] @ coord #[3, 4] @ [4, n]
        img_coord = mc.intrinsic_matrices[cam] @ img_coord
        img_coord[:2, :] /= img_coord[2, :]
        img_coord = img_coord[:2, :].T
        plt.subplot(121)
        plt.imshow(img)
        for c in img_coord:
            plt.scatter(c[0], c[1], s=2, c='red')
        plt.axis('off')
        plt.subplot(122)
        plt.axis('off')
        plt.imshow(depth_map)
        plt.show()