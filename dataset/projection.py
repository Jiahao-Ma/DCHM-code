import numpy as np


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

def get_imagecoord_from_worldcoord3D(world_coord, intrinsic_mat, extrinsic_mat):
    
    if world_coord.shape[0] == 2:
        world_coord = np.concatenate([world_coord, np.zeros([1, world_coord.shape[1]])], axis=0)
    world_coord_hom = np.concatenate([world_coord, np.ones([1, world_coord.shape[1]])], axis=0)
    cam_coord = extrinsic_mat @ world_coord_hom
    depth = cam_coord[2, :]
    img_coord = intrinsic_mat @ cam_coord 
    img_coord = img_coord[:2, :] / img_coord[2, :]
    return img_coord, depth, cam_coord