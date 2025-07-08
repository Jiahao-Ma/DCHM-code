import os, sys; sys.path.append(os.getcwd())
import torch
import numpy as np
from model.ffe.gaussian_filter import GaussianFilter
from lib.utils.tool_utils import to_numpy

def getRay(points, K, c2w, mode='OpenCV'):
    if mode == 'OpenCV':
        weight = [1, 1, 1]
    elif mode == 'CG': # COLMAP, Blender, OpenGL
        weight = [-1, -1, -1]
    else:
        raise ValueError('Unknow camera system mode.')
    dirs = np.stack([  weight[0] * ( points[:, 0] - K[0][2] ) / K[0][0], weight[1] * ( points[:, 1] - K[1][2] ) / K[1][1], weight[2] * np.ones_like(points[:, 0]) ], -1) #(N, 3)
    R_c = c2w[:3, :3] # rotation matrix from **camera** coordinate to **world** coordinate
    rays_d = np.sum(dirs[..., None, :] * R_c, axis=-1) # (N, 3)
    rays_o = np.broadcast_to(c2w[:3, -1], np.shape(rays_d))
    return rays_o, rays_d

def getSpaceFromCamera(points, K, c2w, X=None, Y=None, Z=0, mode='OpenCV'):
    """ Calculate the world coordinates from camera coordinate system via 
        ray tracing.
        Args:
            points: ndarray shape of (N, 2) or (2,)
                    contains the screen coordinate of points [uv]
            K: ndarray shape of (3, 3) intrinsic matrix 
                K = | fx  0  cx |
                    | 0  fy  cy |
                    | 0   0   1 |
            c2w: ndarray shapeo of (3, 4) extrinsic matrix 
                [R | t] = | r11  r12  r13  t1|
                          | r21  r22  r23  t2|
                          | r31  r32  r33  t3|        
            X : number, ndarray, optional
                the X coordinate in **world** coordinates of the target points, dimensions scalar, (N)
            Y : number, ndarray, optional
                the Y coordinate in **world** coordinates of the target points, dimensions scalar, (N)
            Z : number, ndarray, optional
                the Z coordinate in **world** coordinates of the target points, dimensions scalar, (N), default 0  
        Returns:
            points: ndarray shape of (N, 3) or (3,)
                    contains the world coordinate of points [X,Y,Z]          
    """
    points = np.array(points).reshape(-1,2)
    if X is not None:
        assert X.shape[0] == points.shape[0], 'the number of screen points is not equal to that of given points'
        index = 0
        given = X.flatten()
    elif Y is not None:
        assert Y.shape[0] == points.shape[0], 'the number of screen points is not equal to that of given points'
        index = 1
        given = Y.flatten()
    elif Z is not None:
        index = 2
        if Z == 0:
            given = np.zeros_like(points[:, 0])
        else:
            assert Z.shape[0] == points.shape[0], 'the number of screen points is not equal to that of given points'
            given = Z.flatten()
    # calculate origin and direction of ray
    Ray_o, Ray_d = getRay(points, K, to_numpy(c2w), mode=mode)
    
    # R(t) = Ray_o + factor * Ray_d {r(t) = o + t*d}
    factor = ( given - Ray_o[..., index] ) / Ray_d[..., index] #factor: (N,); given: (N,)
    factor = np.full_like(Ray_d[:, 0], factor)
    points = Ray_o + factor[:, None] * Ray_d
    # print('factor: ', factor)
    if mode == 'OpenCV':
        points[factor < 0] = np.nan
        return points
    elif mode == 'CG':
        return points
        
    

# Calculate the depth value of pedestrian 
def getBBoxDepthMap(BottomCenter, TopCenter, BBoxH, BBoxW, intrinsic_mat=None, extrinsic_mat=None, c2w=None, grid_range=None, mode='OpenCV'):
    """
        Calculate the depth map of bounding box area.
        Args:
            BottomCenter: ndarray (N, 2) the bottom center screen coordinate values of bounding box.
            TopCenter: ndarray (N, 2) the top center screen coordinate values of boudning box.
            BBoxH: ndarray (N,) the pixel height of bounding box in **screen** coordinate system.
            BBoxW: ndarray (N,) the pixel width of bounding box in **screen** coordinate system.
            w2c_in: ndarray (3, 3) intrinsic matrix 
            c2w: the projection matrix from world coordinate to camera coordinate, the inverse of w2c
            w2c_ex: ndarray (3, 4) extrinsic matrix
        Return:
            pedestrian_depth_map: the depth map of pedestrians
    """
    def pts2depth(pts_world, extrinsice_matrix):
        pts_world = np.array(pts_world).reshape(-1, 3)
        pts_world_hom = np.zeros(shape=(pts_world.shape[0], 4))
        pts_world_hom[:, :3] = pts_world[:, :3]
        pts_world_hom[:, -1] = np.ones(shape=(pts_world.shape[0]))
        pts_cam = np.dot(extrinsice_matrix, pts_world_hom.T) # (3, 4) @ (4, N)
        return pts_cam[-1, :]

    assert len(BottomCenter) == len(TopCenter), 'The number of bottom should be equal to that of top.'
    BottomCenter = np.array(BottomCenter).reshape(-1, 2)
    TopCenter = np.array(TopCenter).reshape(-1, 2)
    # Calculate the world coordinate value of feet and head.
    feet_pts_world = getSpaceFromCamera(BottomCenter, intrinsic_mat, c2w, Z=0, mode=mode)
    head_pts_world = getSpaceFromCamera(TopCenter, intrinsic_mat, c2w, X=np.array(feet_pts_world[:, 0]).reshape(-1, 1), mode=mode)
    head_pts_world[:, :2] = feet_pts_world[:, :2] # In order to keep pedestrian perpendicular to ground 
    # Calculate the depth value.
    feet_depth = pts2depth(feet_pts_world, extrinsic_mat)
    head_depth = pts2depth(head_pts_world, extrinsic_mat)
    # Construct the depth map of each pedestrian
    pedestrian_depth_map = []
    for i, (H, W) in enumerate(zip(BBoxH, BBoxW)):
        depth_horizontal_line = np.linspace(head_depth[i], feet_depth[i], int(H)).reshape(-1, 1)
        pedestrian_depth_map.append(np.tile(depth_horizontal_line, (1, int(W))))
    if grid_range is not None:
        # judge if the pedestrian is locate within the specific range
        x_min, x_max, y_min, y_max, _, _ = grid_range.cpu().numpy()
        mask = np.logical_or(np.logical_or(feet_pts_world[:, 0] < x_min, feet_pts_world[:, 0] > x_max), \
                            np.logical_or(feet_pts_world[:, 1] < y_min, feet_pts_world[:, 1] > y_max))
        # filter out the pedestrian outside square
        mask = ~mask
        try:
            return np.array(pedestrian_depth_map, dtype=object)[mask], mask
        except:
            return [pedestrian_depth_map[i] for i in range(len(mask)) if mask[i]], mask
    else:
        #keep all pedestrian in square
        mask = torch.arange(feet_pts_world.shape[0]).to(torch.long)
        return np.array(pedestrian_depth_map, dtype=object), mask

def c2w_cvt(w2c):
    assert w2c.shape == (3, 4) or (4, 4), 'the shape of project matric should be (3,4) or (4,4)'
    if w2c.shape != (4, 4):
        w2c = np.vstack([w2c, np.zeros((1, 4), dtype=w2c.dtype)])
        w2c[-1, -1] = 1
    c2w = np.linalg.inv(w2c)
    return c2w[:3, :4]

def fill_depth(depth_map, depth_list, bboxes, pc_filter_op):
    floor_depth_map_mask = depth_map != 0
    filter_mode = list(pc_filter_op.keys())[0]
    # sort depth value
    max_depth_list = [torch.max(depth) for depth in depth_list if np.array(depth.shape).all()]
    depth_idx = np.argsort(max_depth_list)[::-1] # from far to close
    # discard the pedestrian who are not in the range of ground( depth == nan )
    for idx in depth_idx:
        if torch.isnan(max_depth_list[idx]):
            continue
        relative_depth = depth_list[idx]
        bbox = bboxes[idx]
        bbox = [int(bb) for bb in bbox]
        if filter_mode == 'no_filter':
            pass

        elif filter_mode == 'random_filter':
            keep_human_point_cloud_rate = pc_filter_op[filter_mode]['keep_human_point_cloud_rate']
            relative_depth = random_filter_human_pc(relative_depth, keep_human_point_cloud_rate, np.float64('inf'))

        elif filter_mode == 'gaussian_filter':
            sigma = pc_filter_op[filter_mode]['sigma']
            relative_depth = GaussianFilter(relative_depth, sigma)
        try:
            depth_map[bbox[1]:bbox[3], bbox[0]:bbox[2]] = relative_depth
            floor_depth_map_mask[bbox[1]:bbox[3], bbox[0]:bbox[2]] = False
        except:
            H, W = depth_map[bbox[1]:bbox[3], bbox[0]:bbox[2]].shape
            depth_map[bbox[1]:bbox[3], bbox[0]:bbox[2]] = relative_depth[:H, :W]
            floor_depth_map_mask[bbox[1]:bbox[3], bbox[0]:bbox[2]] = False
    floor_depth_map = depth_map.clone()
    floor_depth_map[floor_depth_map_mask] = 0.
    return depth_map, floor_depth_map 
    # depth_map: floor_depth_map + pedestrian_depth_map

def random_filter_human_pc(depth_map, rate, value):
    # randomly remove the humans' point cloud in depth_map, `rate` percent of point clouds are retained,
    # the parts that need to be removed will be filled by 0
    _, W = depth_map.shape
    num_elem = depth_map.numel()
    index = np.arange(num_elem)
    np.random.shuffle(index)
    index = index[int(rate * num_elem):]
    r_idx = index // W
    c_idx = index % W
    depth_map[r_idx, c_idx] = value
    return depth_map



def random_filter_floor_pc(pc, rate):    
    # randomly remove the floor's point cloud in depth_map, `rate` percent of point clouds are retained,
    idx = torch.randperm(pc.shape[0])
    pc = pc[idx]
    pc = pc[:int(rate * pc.shape[0])]
    return pc

if __name__ == '__main__':
    import os, sys; sys.path.append(os.getcwd())
    from lib.data.dataloader_bk import MultiviewDataset, TransformAnnot, ResizeV14, PadV14, UnNormalizer
    
    from torchvision import transforms
    from lib.data.wildtrack import Wildtrack
    from lib.data.multiviewX import MultiviewX
    from lib.utils.tool_utils import to_numpy, make_grid 
    from model.ffe.pcl import floorGrid_to_floorDepthMap
    from PIL import Image
    import matplotlib.pyplot as plt
    old_h, old_w = 1080, 1920
    new_h, new_w = 720, 1280
    scale_h = new_h / old_h
    scale_w = new_w / old_w
    scale = np.array([[scale_w,   0,    0],
                      [  0,   scale_h, 0 ],
                      [  0,      0,    1 ]])
    # pad_h, pad_w = get_padded_value(new_h, new_w)
    pad_h, pad_w = 0, 0
    mean = torch.tensor(np.array([0.485, 0.456, 0.406]), dtype=torch.float32)
    std = torch.tensor(np.array([0.229, 0.224, 0.225]), dtype=torch.float32)
        
    transform_img = transforms.Compose([transforms.Resize(size=(new_h, new_w)),
                                        transforms.ToTensor(),
                                        transforms.Normalize(mean=mean, std=std),
                                        transforms.Pad(padding=(0, 0, pad_w, pad_h)),
                                        ])

    transform_annot = transforms.Compose([ TransformAnnot(scale_h=scale_h,
                                                          scale_w=scale_w) ])
    if float(torch.__version__[:3]) <= 1.4:
        transform_depth = transforms.Compose([ResizeV14(size=(new_h, new_w)),
                                              PadV14(padding=(0, pad_w, 0, pad_h))])                
    else:
        transform_depth = transforms.Compose([transforms.Resize(size=(new_h, new_w)),
                                              transforms.Pad(padding=(0, 0, pad_w, pad_h))])                

    unnormalize = UnNormalizer(mean=mean, std=std)
    DATASET = {'MultiviewX': MultiviewX, 'Wildtrack': Wildtrack}
    dataname = 'Wildtrack'
    path = 'F:\ANU\ENGN8602\Data\{}'.format(dataname) # MultiviewX
    dataset = DATASET[dataname](root=path)
    CAM = 0
    ex_mat = dataset.extrinsic_matrices[CAM] # (3, 4) MATRIC
    in_mat = dataset.intrinsic_matrices[CAM] # (3, 3) MATRIC
    poses_world_grid = dataset.labels_pos[0][CAM+1]
    poses_labels = dataset.labels_bbox[0][CAM+1]
    path = 'F:\\ANU\\ENGN8602\\Data\\Wildtrack\\Image_subsets\\C{}\\00000000.png'.format(CAM+1)
    img = Image.open(path)
    plt.imshow(img)
    plt.axis('off')
    for world_grid, label in zip(poses_world_grid, poses_labels):
        world_coord = DATASET[dataname].get_worldcoord_from_worldgrid(world_grid)
        plt.scatter(label[-2], label[-1], s=3, c='red')
        print('world_coord: ', world_coord)
        print('projected_on_screen:', label[-2:])
        points = getSpaceFromCamera(label[-2:], in_mat, c2w_cvt(ex_mat), Z=0, mode='OpenCV')
        print('reprojected_point:', points)

    plt.show()
