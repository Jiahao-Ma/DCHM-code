import copy, time

# from chardet import detect
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

from lib.utils.visual_utils import Process
from model.ffe.ray import c2w_cvt, getBBoxDepthMap, fill_depth, random_filter_floor_pc
from lib.utils import visual_utils, tool_utils
from model.ffe.pcl import depth_map_to_point_cloud_Tensor, Transform_point_cloud_from_cam_to_world_Tensor, write_point_clouds, bbox_depth_map_to_point_cloud_Tensor
class DepthFFE(nn.Module):
    def __init__(self, 
                 dataset,
                 config, 
                 process:Process,
                 ):
        super().__init__()
        model_cfg = config.MODEL
        self.K = dataset.intrinsic_matrices
        self.w2c = dataset.extrinsic_matrices
        self.c2w = [c2w_cvt(w2c) for w2c in self.w2c]
        self.dataset = dataset.base.__name__

        # used to generate relative depth map
        self.depth_map = dataset.depth_map
        self.depth_max = dataset.depth_max
        try:
            self.depth_min = dataset.depth_min
        except:
            self.depth_min = None
        self.grid_range = dataset.grid_range
        self.norm = dataset.base.norm
        self.world2grid = dataset.base.get_worldgrid_from_worldcoord_Tensor
        self.contain_floor_pc = model_cfg.FFE.CONTAIN_FLOOR_POINT_CLOUD
        if self.contain_floor_pc:
            self.keep_floor_point_cloud_rate = model_cfg.FFE.KEEP_FLOOR_POINT_CLOUD_RATE
        else:
            self.keep_floor_point_cloud_rate = 0

        if model_cfg.FFE.PC_PROCESSOR.NAME == 'gaussian_filter':
            self.pc_filter_op = {'gaussian_filter': model_cfg.FFE.PC_PROCESSOR.OPERATOR['gaussian_filter'] }
            
        elif model_cfg.FFE.PC_PROCESSOR.NAME == 'random_filter':
            self.pc_filter_op = {'random_filter' : model_cfg.FFE.PC_PROCESSOR.OPERATOR['random_filter'] }
            
        elif model_cfg.FFE.PC_PROCESSOR.NAME == 'no_filter':
            self.pc_filter_op = {'no_filter': None } 

        self.transform_mode = model_cfg.FFE.TRANSFORM
        self.unnormalize = process.reverse_image_feat
        self.normalize = process.normalize_image_PIL


    def create_depth_map(self, bboxes, pred_keypoints, cam, image_size, use_btm_ctn_as_feet=False, FIXED_HEIGHT=120):    
        """
            Generate the depth map via the prediction of detector. We define middle center
            of bounding box is the position of pedestrians' feet and top center of bounding
            box is the that of head by default.
            Args:
                bboxes: prediction of detector containing the left top and right bottom of bbox
                pred_keypoints: the estimated keypoint in each bounding box
                cam: camera index
                image_size: (H, W) the size of RGB image
            Returns:
                depth_map: the depth map of RGB image
                mask: used to determine if the pedestrians are within square (detection range)
        """
        K, w2c, c2w = self.K[cam], self.w2c[cam], self.c2w[cam]
        feet_uv, head_uv, BBoxH, BBoxW, bboxes_ = [], [], [], [], []
        depth_map, depth_max = copy.deepcopy(self.depth_map[cam]), self.depth_max[cam]
        try:
            depth_min = self.depth_min[cam]
        except:
            pass
        for bbox, keypoint in zip(bboxes, pred_keypoints):
            if use_btm_ctn_as_feet:
                height = int(bbox[3]) - int(bbox[1])
                width = int(bbox[2]) - int(bbox[0]) 
                bboxes_.append(bbox[:4])
                feet_uv.append([(bbox[0] + bbox[2])/2, bbox[3]])
                head_uv.append([(bbox[0] + bbox[2])/2, bbox[1]])
                BBoxH.append(height)
                BBoxW.append(width)
            else:
                # if the keypoint exceed the figure, adjust the keypoint 
                # 2022/10/22 ignoring this operation, this operation will have impact on the depth calculation
                # keypoint[0] = np.minimum(keypoint[0], image_size[1])
                # keypoint[1] = np.minimum(keypoint[1], image_size[0])
         
                bbox[3] = int(keypoint[1])
                width = int(bbox[2]) - int(bbox[0])
                height = int(bbox[3]) - int(bbox[1])
                if height <= 1:
                    bbox[1] = max(bbox[3] - FIXED_HEIGHT, 0)
                    height = FIXED_HEIGHT
                bboxes_.append(bbox[:4])
                feet_uv.append(keypoint)
                head_uv.append([keypoint[0], bbox[1]]) # feet x, bbox top bottom y
                BBoxH.append(height)
                BBoxW.append(width)

        
        feet_uv = np.array(feet_uv)
        head_uv = np.array(head_uv)
        bboxes_ = np.array(bboxes_)
        grid_range = self.grid_range[cam]

        if self.dataset == 'Wildtrack':
            mode = 'OpenCV'
        elif self.dataset == 'MultiviewX':
            mode = 'CG'
        BBoxH = [float(h) for h in BBoxH]
        # generate per box depth map and a mask to filter out the ground region.
        pedestrians_depth_maps, mask = getBBoxDepthMap(feet_uv, head_uv, BBoxH, BBoxW, K, w2c, c2w, grid_range=grid_range, mode=mode)

        if self.norm == 'max_norm':
            relative_depth = [torch.Tensor(depth.astype(np.float32)) / depth_max * 255 for depth in pedestrians_depth_maps]
        elif self.norm == 'max_min_norm':
            relative_depth = [255 * (torch.Tensor(depth.astype(np.float32)) - depth_min) / (depth_max - depth_min) for depth in pedestrians_depth_maps]
        if self.contain_floor_pc:
            floor_depth_map = depth_map
        else:
            floor_depth_map = torch.full_like(depth_map, torch.tensor(np.float64('inf')))

        depth_map, floor_depth_map = fill_depth(floor_depth_map, relative_depth, bboxes_[mask], self.pc_filter_op)
        
        depth_map = torch.where(torch.isinf(depth_map), torch.zeros_like(depth_map), depth_map)
        floor_depth_map = torch.where(torch.isinf(floor_depth_map), torch.zeros_like(floor_depth_map), depth_map)
        
        relative_depth = [ torch.where(torch.isinf(rd), torch.zeros_like(rd), rd) for rd in relative_depth ]
        return depth_map, mask, bboxes_, relative_depth, floor_depth_map
        # relative_depth has been filtered, bboxes_ hasn't been filtered, mask contain the bool value to determine the validaty of bbox

    def viz_detector_pred(self, images, post_pred_boxes, pred_keypoints, gt_keypoints, masks, depth_maps, H=64, W=48):
        img = visual_utils.reverse_image(images.cpu())
        fig = plt.figure(figsize=(15, 8))
        axes = fig.subplots(1, 2)
        axes[0].imshow(img)
        axes[0].axis('off')
        axes[0] = visual_utils.draw_boxes(
                img=img,
                boxes=post_pred_boxes[:, :4],
                pred_keypoints=pred_keypoints,
                gt_keypoints=gt_keypoints,
                scores=None, #post_pred_boxes[:, 4],
                tags=['Person']*len(post_pred_boxes),
                line_thick=1, line_color='black', 
                ax=axes[0], mask=masks)
        
        axes[1].imshow(tool_utils.to_numpy(depth_maps.cpu()))
        axes[1].axis('off')
        plt.show()

    def forward(self, batch_dict, save_pc=False, viz=False):  
        batch_dict['depth_map_list'] = list(); batch_dict['pedestrian_within_square_mask_list']=list()
        merged_point_cloud_in_world_coord = list()
        # create_depth_map_time = 0
        # depth_to_point_time = 0
        # point_cam_to_world = 0
        # t1 = time.time()
        for cam, (bboxes, image, pred_keypoints) in enumerate(zip(batch_dict['pred_boxes'], batch_dict['images'], batch_dict['kyps_align'])):
            
            # image = torch.zeros_like(image) # TODO: full black
            # image = torch.stack([torch.full_like(image[0], 2.6400), torch.full_like(image[0], 2.4286), torch.full_like(image[0], 2.2489)]) # TODO: full white
            # generate box depth map
            depth_map, pedestrian_within_square_mask, bboxes, relative_depth, floor_depth_map = self.create_depth_map(bboxes, pred_keypoints.squeeze(1), cam, batch_dict['image_size'], use_btm_ctn_as_feet=False)
            batch_dict['depth_map_list'].append(depth_map)
            batch_dict['pedestrian_within_square_mask_list'].append(pedestrian_within_square_mask)
            # ---- Create point cloud ---- #      
            if self.depth_min is not None:
                depth_min = self.depth_min[cam]
            else:
                depth_min = None
     
            if self.transform_mode == 'scene_depth_map_to_point_cloud':
                feature_input = image.permute(1, 2, 0)
                point_clouds_in_cam_coord = depth_map_to_point_cloud_Tensor(rgb_img=feature_input,
                                                                            depth_map=depth_map.to(device=image.device),
                                                                            depth_max=self.depth_max[cam],
                                                                            depth_min=depth_min,
                                                                            K=torch.Tensor(self.K[cam]).to(device=image.device),
                                                                            norm=self.norm)
            elif self.transform_mode == 'bbox_depth_map_to_point_cloud':
         
                feature_input = image.permute(1, 2, 0)
                point_clouds_in_cam_coord = bbox_depth_map_to_point_cloud_Tensor(rgb_img=feature_input,
                                                                                 bbox_depth_map=relative_depth, 
                                                                                 bboxes=bboxes,
                                                                                 mask=pedestrian_within_square_mask,
                                                                                 depth_max=self.depth_max[cam],
                                                                                 depth_min=depth_min, 
                                                                                 K=torch.Tensor(self.K[cam]).to(device=image.device),
                                                                                 norm=self.norm,
                                                                                 )
               
       
                if self.contain_floor_pc:
                    floor_point_cloud_in_cam_coord = depth_map_to_point_cloud_Tensor(rgb_img=image.permute(1, 2, 0),
                                                                                     depth_map=floor_depth_map.to(device=image.device),
                                                                                     depth_max=self.depth_max[cam],
                                                                                     depth_min=depth_min,
                                                                                     K=torch.Tensor(self.K[cam]).to(device=image.device),
                                                                                     norm=self.norm,
                                                                                     )   
                    floor_point_cloud_in_cam_coord = random_filter_floor_pc(floor_point_cloud_in_cam_coord, rate=self.keep_floor_point_cloud_rate)                                                                                     
                    point_clouds_in_cam_coord = torch.cat([point_clouds_in_cam_coord, floor_point_cloud_in_cam_coord], dim=0)  

                                                                                                                                                              
            
            point_clouds_in_world_coord = Transform_point_cloud_from_cam_to_world_Tensor(point_cloud=point_clouds_in_cam_coord, 
                                                                                         project_mat=torch.Tensor(self.c2w[cam]).to(device=point_clouds_in_cam_coord.device))                                                                       
            merged_point_cloud_in_world_coord.append(point_clouds_in_world_coord)
        
            # if batch_dict['images'] is not None and viz:
            #     self.viz_detector_pred(batch_dict['ori_img'][cam], bboxes, batch_dict['kyps_align'][cam], keypoints.cpu(), pedestrian_within_square_mask, depth_map)                                                                                         
        

        merged_point_cloud_in_world_coord = torch.cat(merged_point_cloud_in_world_coord, dim=0)
        merged_point_cloud_in_world_coord[:, :3] = self.world2grid(merged_point_cloud_in_world_coord[:, :3]) # transform points from world to grid
        batch_dict['point_clouds'] = merged_point_cloud_in_world_coord
  
        if save_pc:
            write_point_clouds('./visualization/pc/{}.ply'.format(37), merged_point_cloud_in_world_coord.cpu().numpy())
            print('`visualization/pc/{}.ply`   has been save!'.format(37))
        return batch_dict
