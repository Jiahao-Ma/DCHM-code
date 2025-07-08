import torch
import numpy as np
import copy, sys, os
sys.path.append('/home/dzc/Projects/MvCHM')
import matplotlib.pyplot as plt
from torchvision import transforms
from torch.utils.data import DataLoader
import cv2
import torch.nn.functional as F
from PIL import Image


def fill_depth(depth_map, depth_list, bboxes):
        # sort depth value
        max_depth_list = [np.max(depth) for depth in depth_list]
        depth_idx = np.argsort(max_depth_list)[::-1] # from far to close
        for idx in depth_idx:
            if np.isnan(max_depth_list[idx]):
                continue
            relative_depth = depth_list[idx]
            bbox = bboxes[idx]
            bbox = [int(bb) for bb in bbox]
            depth_map[bbox[1]:bbox[3], bbox[0]:bbox[2]] = relative_depth
        return depth_map

def Transform_point_cloud_from_cam_to_world(point_cloud, project_mat):
    """ --- Numpy Version ---
        Args:
            point_cloud: pts in camera coordinate system numpy.ndarray
                         with the shape of [N, 6] [x, y, z, r, g, b]
            project_mat: projection matrix numpy.ndarray with the shape of [3, 4] [R | T] 
        Return:
            point_cloud: pts in world coordinate system 
                         numpy.ndarray with the shape of [N, ^]
    """
    # project_mat = np.vstack([project_mat, np.zeros((1, 4), dtype=project_mat.dtype)])
    # project_mat[-1, -1] = 1
    # project_mat_inv = np.linalg.inv(project_mat) 
    point_cloud_hom = np.zeros((point_cloud.shape[0], 4))
    point_cloud_hom[:, -1] = 1
    point_cloud_hom[:, :3] = point_cloud[:, :3] # (N, 4)
    # (3, N) = (3, 4) @ (4, N)
    temp = project_mat @  point_cloud_hom.T
    # (4, N) -> (N, 4) -> assign (N, 3)
    point_cloud[:, :3] = (temp.T)[:, :3]
    return point_cloud

def Transform_point_cloud_from_cam_to_world_Tensor(point_cloud, project_mat):
    """ --- Tensor Version ---
        Args:
            point_cloud: pts in camera coordinate system numpy.ndarray
                         with the shape of [N, 6] [x, y, z, r, g, b]
            project_mat: projection matrix numpy.ndarray with the shape of [3, 4] [R | T] 
        Return:
            point_cloud: pts in world coordinate system 
                         numpy.ndarray with the shape of [N, ^]
    """
    point_cloud_hom = torch.zeros((point_cloud.shape[0], 4), device=point_cloud.device)
    point_cloud_hom[:, -1] = 1
    point_cloud_hom[:, :3] = point_cloud[:, :3] # (N, 4)
    # (4, N) = (4, 4) @ (4, N)
    temp = project_mat @  point_cloud_hom.T
    # (4, N) -> (N, 4) -> assign (N, 3)
    point_cloud[:, :3] = (temp.T)[:, :3]
    return point_cloud    

def depth_map_to_point_cloud(rgb_img=None, depth_map=None, depth_max=None, depth_min=None, K=None, norm='max_norm'):
    """ --- Numpy Version ---
        Args:
            rgb_img: `np.ndarray` an array shape of [H, W, C]
            depth_map: `np.ndarray` an array shape of [H, W] storing depth value of each pixel
            depth_max: the maximum of depth map
            K: intrinsic matrix 
               K = [[fx, 0, cx],
                    [0, fy, cy],
                    [0,  0, 1]]
        Returns:
            pts_in_cam: ndarray: [X, Y, Z, R, G, B] point clouds in local camera coordinate system. 
    """
    assert norm in ['max_norm', 'max_min_norm'], 'Unknown normalization'
    if norm == 'max_min_norm':
        assert depth_min != None, 'depth_min can not be empty in `max_min_norm` mode.'
    H, W, _ = rgb_img.shape
    v = np.arange(0, H)
    u = np.arange(0, W)
    u, v = np.meshgrid(u, v)
    u, v = u.astype(np.float32), v.astype(np.float32)
    if norm == 'max_norm':
        Z = depth_map.astype(np.float32) / 255 * depth_max
        X = Z * (u - K[0,2]) / K[0, 0]
        Y = Z * (v - K[1, 2]) / K[1, 1]
        valid = Z > 0 # all valid depth value in wildtrack dataset are positive
        pts_in_cam = np.concatenate([X[:, :, None], Y[:, :, None], Z[:, :, None], rgb_img], axis=-1)
        pts_in_cam = pts_in_cam[valid]
        return pts_in_cam
    elif norm == 'max_min_norm':
        Z = ( depth_map.astype(np.float32) / 255 ) * (depth_max - depth_min) + depth_min
        X = Z * (u - K[0,2]) / K[0, 0]
        Y = Z * (v - K[1, 2]) / K[1, 1]
        valid = Z < 0 # all valid depth value in multiviewX dataset are negative
        pts_in_cam = np.concatenate([X[:, :, None], Y[:, :, None], Z[:, :, None], rgb_img], axis=-1)
        # pts_in_cam = pts_in_cam.reshape(-1, 6)
        pts_in_cam = pts_in_cam[valid]
        return pts_in_cam
    
def depth_map_to_point_cloud_Tensor(rgb_img=None, depth_map=None, depth_max=None, depth_min=None, K=None, norm='max_norm', proj=None):
    """ --- Tensor Version ---
        Args:
            rgb_img: `np.ndarray` an array shape of [H, W, C]
            depth_map: `np.ndarray` an array shape of [H, W] storing depth value of each pixel
            depth_max: the maximum of depth map
            K: intrinsic matrix 
               K = [[fx, 0, cx],
                    [0, fy, cy],
                    [0,  0, 1]]
        Returns:
            pts_in_cam: ndarray: [X, Y, Z, R, G, B] point clouds in local camera coordinate system. 
    """
    assert norm in ['max_norm', 'max_min_norm'], 'Unknown normalization'
    if norm == 'max_min_norm':
        assert depth_min != None, 'depth_min can not be empty in `max_min_norm` mode.'
    H, W, _ = rgb_img.shape
    v = torch.arange(0, H).to(device=rgb_img.device)
    u = torch.arange(0, W).to(device=rgb_img.device)
    u, v = torch.meshgrid(u, v)
    u, v = u.to(torch.float32).t(), v.to(torch.float32).t()
    if norm == 'max_norm':
        Z = depth_map.to(torch.float32) / 255 * depth_max
        X = Z * (u - K[0, 2]) / K[0, 0]
        Y = Z * (v - K[1, 2]) / K[1, 1]
        valid = Z > 0
        pts_in_cam = torch.cat([X[:, :, None], Y[:, :, None], Z[:, :, None], rgb_img], dim=-1)
        pts_in_cam = pts_in_cam[valid]
        return pts_in_cam
    elif norm == 'max_min_norm':
        Z = ( depth_map.to(torch.float32) / 255 ) * (depth_max - depth_min) + depth_min
        X = Z * (u - K[0,2]) / K[0, 0]
        Y = Z * (v - K[1, 2]) / K[1, 1]
        valid = Z < 0
        pts_in_cam = torch.cat([X[:, :, None], Y[:, :, None], Z[:, :, None], rgb_img], dim=-1)
        pts_in_cam = pts_in_cam[valid]
        return pts_in_cam

def bbox_depth_map_to_point_cloud_Tensor(rgb_img=None, bbox_depth_map=None, bboxes=None, mask=None, depth_max=None, depth_min=None, K=None, norm='max_norm'):
    assert norm in ['max_norm', 'max_min_norm'], 'Unknown normalization'
    if norm == 'max_min_norm':
        assert depth_min != None, 'depth_min can not be empty in `max_min_norm` mode.'
    H, W, _ = rgb_img.shape
    v = torch.arange(0, H).to(device=rgb_img.device)
    u = torch.arange(0, W).to(device=rgb_img.device)
    u, v = torch.meshgrid(u, v)
    u, v = u.to(torch.float32).t(), v.to(torch.float32).t()
    bboxes = bboxes[mask]
    pts_in_cam_list = list()
    if norm == 'max_norm':
        for dp, bb in zip(bbox_depth_map, bboxes):
            bb = [int(b) for b in bb]
            u_ = u[bb[1]:bb[3], bb[0]:bb[2]]
            v_ = v[bb[1]:bb[3], bb[0]:bb[2]]
            Z = dp.to(dtype=torch.float32, device=rgb_img.device) / 255  * depth_max
            if u_.shape != Z.shape:
                Z = Z[:u_.shape[0], :u_.shape[1]]
            X = Z * (u_ - K[0, 2]) / K[0, 0]
            Y = Z * (v_ - K[1, 2]) / K[1, 1]
            valid = Z > 0 
            pts_in_cam = torch.cat([X[:, :, None], Y[:, :, None], Z[:, :, None], rgb_img[bb[1]:bb[3], bb[0]:bb[2]]], dim=-1)
            pts_in_cam = pts_in_cam[valid]
            pts_in_cam_list.append(pts_in_cam)
    elif norm == 'max_min_norm':
        for dp, bb in zip(bbox_depth_map, bboxes):
            bb = [int(b) for b in bb]
            u_ = u[bb[1]:bb[3], bb[0]:bb[2]]
            v_ = v[bb[1]:bb[3], bb[0]:bb[2]]
            Z = ( dp.to(dtype=torch.float32, device=rgb_img.device) / 255 ) * (depth_max - depth_min) + depth_min
            if u_.shape != Z.shape:
                Z = Z[:u_.shape[0], :u_.shape[1]]
            X = Z * (u_ - K[0, 2]) / K[0, 0]
            Y = Z * (v_ - K[1, 2]) / K[1, 1]
            valid = Z < 0 
            pts_in_cam = torch.cat([X[:, :, None], Y[:, :, None], Z[:, :, None], rgb_img[bb[1]:bb[3], bb[0]:bb[2]]], dim=-1)
            pts_in_cam = pts_in_cam[valid]
            pts_in_cam_list.append(pts_in_cam) # TODO: figure out the `inf` value
    pts_in_cam_list = torch.cat(pts_in_cam_list, dim=0)
    return pts_in_cam_list


def write_point_clouds(ply_filename, points):
    if not os.path.exists(os.path.dirname(ply_filename)):
        os.makedirs(os.path.dirname(ply_filename))
    formatted_points = []
    for point in points:
        formatted_points.append("%f %f %f %d %d %d 0\n" % (point[0], point[1], point[2], point[5], point[4], point[3])) # blue <-> red for CloudCompare visualization

    out_file = open(ply_filename, "w")
    out_file.write('''ply
    format ascii 1.0
    element vertex %d
    property float x
    property float y
    property float z
    property uchar blue
    property uchar green
    property uchar red
    property uchar alpha
    end_header
    %s
    ''' % (len(points), "".join(formatted_points)))
    out_file.close()

def read_point_clouds(ply_filename):
    with open(ply_filename, 'r') as f:
        pts = f.readlines()[12:]
        pts = [line.strip().replace('\n', '')for line in pts]
        pts = np.array([float(p) for pt in pts if pt !='' for p in pt.split(' ')]).reshape(-1, 7)
    return pts[:, :6]

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

def sparse_depth_map(Pts, H, W, normalize=True):
    depth_map = np.zeros((H, W), dtype=np.int32)
    depth_map[np.int32(Pts[1]), np.int32(Pts[0])] = np.int32(Pts[2])
    if normalize:
        max_depth = np.max(depth_map)
        depth_map = depth_map / np.max(depth_map) * 255
        return depth_map, max_depth
    return depth_map

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

if __name__ == "__main__":
    import torch.nn as nn
    from lib.data.wildtrack import Wildtrack 
    from lib.data.multiviewX import MultiviewX
    from lib.utils.tool_utils import to_numpy, make_grid
    # from DEEP.models.retinanet import RetinaNet_resnet50
    from lib.config.wt_config import wt_opts as opts
    from lib.data.dataloader_bk import MultiviewDataset, collater, TransformAnnot, UnNormalizer
    from lib.utils.config_utils import cfg, cfg_from_yaml_file
    from model.ffe.ray import getBBoxDepthMap, c2w_cvt
    import matplotlib; #matplotlib.use('TkAgg')
    from lib.utils.tool_utils import project

    def point_cloud_to_depth_map(ax, bboxes, poses, dataset, cam, HW, depth_map=None, depth_max=None, depth_min=None, linewidth=1, draw_depth=True, draw_box=True, norm='max_norm'):
        # get projection matrix and its inverse
        K = dataset.intrinsic_matrices[cam] # intrinsic matrix
        w2c = dataset.extrinsic_matrices[cam] # extrinsic matrix
        dataname = dataset.base.__name__
        c2w = c2w_cvt(w2c) # the inverse of extrinsic matrix
        if depth_map is not None:
            depth_map[depth_map==0] = np.float64('inf')
        floor_depth = copy.copy(depth_map)
        feet_uv, head_uv, BBoxH, BBoxW, bboxes_ = [], [], [], [], []
        for bbox, pos in zip(bboxes, poses):
            [xmin, ymin, xmax, ymax] = bbox[:4]
            if xmin == -1 and ymin == -1 \
                and xmax == -1 and ymax == -1:
                continue
            xmin = np.maximum(xmin, 0)
            ymin = np.maximum(ymin, 0)
            xmax = np.minimum(xmax, HW[1])
            ymax = np.minimum(ymax, HW[0])
            width = xmax - xmin
            height = ymax - ymin
            coord = dataset.base.get_worldcoord_from_worldgrid(pos)
            pts = project(K @ w2c, coord)

            feet_uv.append([pts[0], ymax])
            head_uv.append([pts[0], ymin])
            BBoxH.append(height)
            BBoxW.append(width)
            bboxes_.append([xmin, ymin, xmax, ymax])
            if draw_box:
                rect = plt.Rectangle([xmin, ymin], width, height, color=(1, 0, 0), linewidth=linewidth, fill=False)
                ax.add_patch(rect)
        if draw_depth:
            feet_uv = np.array(feet_uv)
            head_uv = np.array(head_uv)
            mode = 'OpenCV' if dataname == 'Wildtrack' else 'CG'
            pedestrians_depth_maps, mask = getBBoxDepthMap(feet_uv, head_uv, BBoxH, BBoxW, K, w2c, c2w, mode=mode)
            if norm == 'max_norm':
                relative_depth = [depth / depth_max * 255 for depth in pedestrians_depth_maps]
            elif norm == 'max_min_norm':
                relative_depth = [(depth - depth_min) / (depth_max - depth_min) * 255 for depth in pedestrians_depth_maps]
            depth_map = fill_depth(depth_map, relative_depth, bboxes_)
            depth_map[depth_map==np.float64('inf')] = np.float64(0)
            return ax, depth_map, pedestrians_depth_maps, floor_depth
        else:
            return ax

    def save_image(image, save_dir):
        plt.figure(figsize=(15, 8))
        plt.axis('off')
        plt.imshow(image)
        plt.savefig(save_dir, bbox_inches='tight', pad_inches=0, dpi=300)
        print('Image has been saved in ', save_dir)
        plt.close()
    
    def vis_depth_map(image=None, annots=None, dataset=None, cam=None, depth_map_floor=None, 
                      depth_max=None, depth_min=None, save=False, random_skip=False, 
                      norm='max_norm', dataname='Wildtrack'):
        if random_skip:
            random_skip_num = np.random.randint(3)
            num_annot = len(annots['annot'][cam])
            mask = np.random.choice(num_annot, num_annot - random_skip_num)
            annots['annot'][cam] = annots['annot'][cam][mask]
        fig = plt.figure(figsize=(15, 8))
        axes = fig.subplots(1, 2)
        axes[0].imshow(image)
        axes[0] = point_cloud_to_depth_map(ax=axes[0], 
                                           bboxes=annots['annot'][cam],
                                           poses=annots['pos'][cam],
                                           dataset=dataset, 
                                           cam=cam, 
                                           HW=dataset.img_size,
                                           linewidth=2, 
                                           draw_depth=False, 
                                           norm=norm)
        axes[0].axis('off')
        # plt.savefig('figures\predict\C%d_cam.png'%cam, bbox_inches = 'tight', pad_inches=0, dpi=300)

        axes[1], depth_map, pedestrian_depth, floor_depth = point_cloud_to_depth_map(ax=axes[1], 
                                                                                     bboxes=annots['annot'][cam],
                                                                                     poses=annots['pos'][cam],
                                                                                     dataset=dataset, 
                                                                                     cam=cam, 
                                                                                     HW=dataset.img_size, 
                                                                                     depth_map=depth_map_floor,
                                                                                     depth_max=depth_max,
                                                                                     depth_min=depth_min, 
                                                                                     linewidth=0.5, 
                                                                                     draw_box=False, 
                                                                                     norm=norm)
        axes[1].imshow(depth_map) 
        
        axes[1].axis('off')
        # plt.savefig('figures\predict\C%d_depth.png'%cam, bbox_inches = 'tight', pad_inches=0, dpi=300)
        if save:
            if not os.path.exists('visualization/%s/predict'%dataname):
                os.makedirs('visualization/%s/predict'%dataname)
            if not os.path.exists('visualization/%s/combination'%dataname):
                os.makedirs('visualization/%s/combination'%dataname)
            plt.savefig('visualization/%s/predict/C%d.png'%(dataname, cam), bbox_inches = 'tight', pad_inches=0, dpi=300)
            plt.close()
            # save_image(depth_map, 'visualization\{}\predict\C{}_depth.png'.format(dataname, cam))
            # save_image(pedestrian_depth, 'visualization\{}\combination\C{}_pedestrians_depth.png'.format(dataname, cam))
            # save_image(floor_depth, 'visualization\{}\combination\C{}_floor_depth.png'.format(dataname, cam))
        else:
            plt.show()
        return depth_map, pedestrian_depth, floor_depth

    # CHOOSE YOUR DATASET!
    dataname = 'Wildtrack'
    depth_min = None
    
    DATASET = {'MultiviewX': MultiviewX, 'Wildtrack': Wildtrack}
    root = '/home/dzc/Data/{}'.format(dataname) # the Path of Wildtrack dataset
    # world_size = (480, 1440) # width and length of designed grid - Wildtrack
    world_size = (640, 1000) # width and length of designed grid - MultiviewX
    unnormalize = UnNormalizer()
    pad_w = 32 - 1920 % 32
    pad_h = 32 - 1080 % 32
    transform_img = transforms.Compose([transforms.Pad(padding=(0, 0, pad_w, pad_h)),
                                        transforms.ToTensor(),
                                        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                            std=[0.229, 0.224, 0.225])])

    dataset = MultiviewDataset( DATASET[dataname](root=root), 
                                      set_name='val', 
                                      transform_img=transform_img,
                                      transform_annot=None,
                                      transform_depth=None )
    # val_loader = DataLoader( val_data, num_workers=1, collate_fn = collater )

    # retinanet = RetinaNet_resnet50( num_classes=80 )
    # use_gpu = True

    # if use_gpu:
    #     if torch.cuda.is_available():      
    #         retinanet.load_state_dict(torch.load(r'DEEP\models\checkpoint\coco_resnet_50_map_0_335_state_dict.pt'), strict=False)
    #         retinanet = retinanet.cuda()

    sample = next(iter(dataset))
    intrinsics, extrinsics = dataset.intrinsic_matrices, dataset.extrinsic_matrices
    grid = to_numpy(make_grid(world_size=world_size, cube_LW=[1,1])).reshape(-1, 3)
    depth_map_save_root = os.path.join('/home/dzc/Projects/MvCHM/lib/data/depth_map', dataset.base.__name__)
    point_cloud_world_list = list()
    if not os.path.exists(depth_map_save_root):
        os.mkdir(depth_map_save_root)
    for cam in range(0, dataset.base.num_cam):
        depth_map = np.array(dataset.depth_map[cam])
        depth_max = np.array(dataset.depth_max[cam])
        if dataname == 'Wildtrack':
            depth_min = None
        elif dataname == 'MultiviewX':
            depth_min = np.array(dataset.depth_min[cam])
        # original image visualization
        rgb_img = np.array(255 * unnormalize(sample['img'][cam])).copy()
        rgb_img = np.clip(rgb_img, a_min=0, a_max=255)
        rgb_img = np.transpose(rgb_img, (1, 2, 0))
        rgb_img = rgb_img.astype(np.uint8)
        depth_map, _, _ = vis_depth_map(image=rgb_img[:1080, :1920],
                                        annots=sample, 
                                        dataset=dataset,
                                        cam=cam, 
                                        depth_map_floor=depth_map, 
                                        depth_max=depth_max, 
                                        depth_min=depth_min, 
                                        save=False, 
                                        random_skip=False, 
                                        norm=DATASET[dataname].norm,
                                        dataname=dataname)

        point_cloud = depth_map_to_point_cloud( rgb_img=rgb_img[:1080, :1920], 
                                                depth_map=depth_map,
                                                depth_max=depth_max, 
                                                depth_min=depth_min,
                                                K=intrinsics[cam], 
                                                norm=DATASET[dataname].norm )

        # write_point_clouds('visualization\\%s\\rec_point_cloud\\c%d_point_cloud.ply'%(dataname, cam), point_cloud) 
        # point_cloud = read_point_clouds('visualization\\%s\\rec_point_cloud\\c%d_point_cloud.ply'%(dataname, cam))
        point_cloud_world = Transform_point_cloud_from_cam_to_world(point_cloud, c2w_cvt(extrinsics[cam]))
    
        point_cloud_world_list.extend(point_cloud_world)
    write_point_clouds('visualization/%s/point_cloud_v1.ply'%(dataname), point_cloud_world_list) 
