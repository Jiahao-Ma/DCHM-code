import os
import cv2
import torch
import time
import numpy as np
from tqdm import tqdm
import open3d as o3d
import scipy.ndimage as ndi
from argparse import ArgumentParser
from sklearn.decomposition import PCA
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from dataset.wildtrack import Wildtrack, SuperPixels, WildtrackDetection, frameDataset
from submodules.gsplat.strategy import DefaultStrategy
from ProteusLib import SuperPixelGaussian, render, depth_constraints, DisparityToDepth, pearson_correlation_loss, init_splats_from_ply, rasterize_splats, init_splats_from_pth, rasterize_splats_id
from s04_gs_us import bg_filter_fn
from sklearn.cluster import DBSCAN
from utils.evaluation.evaluate import evaluate_rcll_prec_moda_modp

def filterGS(splats, max_num_gs=50_0000):
    num_splats = splats['means'].size(0)
    if num_splats > max_num_gs:
        print(f'Filter the splats from {num_splats} to {max_num_gs}')
        idx = np.random.choice(num_splats, max_num_gs, replace=False)
        for key in splats.keys():
            splats[key] = splats[key][idx]
    return splats

def clusterGS(args, save_frame=None):
    print(args)
    data_root = args.root

    start_with = args.start_with # start to process from `start_with` th frame
    end_with = args.end_with #-1 # end to process at `end_with` th frame
    print(f'The start_with and end_with are: {start_with} and {end_with}')
    wildtrack_data = Wildtrack(data_root, mask_type='people', mask_label_type='split', start_with=args.start_with, end_with=args.end_with)
    wildtrack_dataloader = DataLoader(wildtrack_data, batch_size=1, pin_memory=True, num_workers=4, drop_last=True)
    sp_dataset = SuperPixels(data_root, n_segments=30, HW=wildtrack_data.image_feat_size, start_with=args.start_with, end_with=args.end_with)
    for data_idx, data in enumerate(wildtrack_dataloader):
        t0 = time.time()
        H, W = data['H'].item(), data['W'].item()
        frame_idx = data['frame_idx'][0]
        save_folder = os.path.join(data_root, 'depths', frame_idx, 'gs_optimize')
        data_folder = os.path.join(data_root, 'depths', frame_idx, 'gs_optimize')
        ckpt = os.path.join(data_folder, f'gs_{args.round}.pth')
        assert os.path.exists(ckpt), f'Fail to find {ckpt}'
        print('Load checkpoint from', ckpt)
        splats = torch.load(ckpt)['splats']
        splats = filterGS(splats=splats, max_num_gs=400_000)
        splats_human_ids = np.zeros((splats['means'].size(0),), dtype=np.int32)
        masks = data['mask'].squeeze(0).cuda()
        gt_mask_comb_b = torch.where(masks > 0, torch.ones_like(masks), torch.zeros_like(masks)) # (7, h, w)
        gt_mask_split = masks.cpu().numpy().astype(np.int32)
    
        gt_mask_comb_b = gt_mask_comb_b.bool().unsqueeze(-1)
        gt_mask_comb_b_np = gt_mask_comb_b.squeeze(-1).cpu().numpy()
        # gt_images = (data['nonnorm_image'].squeeze(0).cuda() / 255.0).permute(0, 2, 3, 1)

        c2ws = torch.cat([cam['c2w']for cam in  data['novel_view']], dim=0).float().cuda()
        w2cs = torch.cat([cam['w2c'] for cam in data['novel_view']], dim=0).float().cuda()
        Ks = torch.cat([cam['K'] for cam in data['novel_view']], dim=0).float().cuda()
        width = data['novel_view'][0]['width'].item()
        height = data['novel_view'][0]['height'].item()
        

        weights_threshold = 0.05
        propagated_human_id_masks = []
        cur_human_id = 0
        for cam_idx in range(len(data['novel_view'])):
            if cam_idx  == 0:
                weights, gs_ids, pixel_ids, camera_ids, _ = rasterize_splats_id(splats, w2cs[cam_idx:cam_idx+1], Ks[cam_idx:cam_idx+1], width, height)
                valid_mask = weights > weights_threshold
                gs_ids = gs_ids[valid_mask].cpu().numpy().astype(np.int32)
                pixel_ids = pixel_ids[valid_mask].cpu().numpy()
                camera_ids = camera_ids[valid_mask].cpu().numpy()
                weights = weights[valid_mask]
                pixel_uv = np.stack([pixel_ids % width, pixel_ids // width], axis=1).astype(np.int32)
                gs_human_ids = gt_mask_split[cam_idx, pixel_uv[:, 1], pixel_uv[:, 0]]
                gs_ids = gs_ids[gs_human_ids!=0]
                gs_human_ids = gs_human_ids[gs_human_ids!=0]
                splats_human_ids[gs_ids] = gs_human_ids
                propagated_human_id_masks.append(gt_mask_split[cam_idx])
                cur_human_id = propagated_human_id_masks[-1].max()
                # plt.figure(figsize=(20, 10))
                # plt.imshow(propagated_human_id_masks[-1])
                # plt.show()
            else:
                weights, gs_ids, pixel_ids, camera_ids, _ = rasterize_splats_id(splats, w2cs[cam_idx:cam_idx+1], Ks[cam_idx:cam_idx+1], width, height)
                valid_mask = weights > weights_threshold
                gs_ids = gs_ids[valid_mask].cpu().numpy().astype(np.int32)
                pixel_ids = pixel_ids[valid_mask].cpu().numpy()
                camera_ids = camera_ids[valid_mask].cpu().numpy()
                weights = weights[valid_mask]
                pixel_uv = np.stack([pixel_ids % width, pixel_ids // width], axis=1).astype(np.int32)
                propagated_human_id_mask = np.zeros_like(gt_mask_split[cam_idx])
                propagated_human_id_mask[pixel_uv[:, 1], pixel_uv[:, 0]] = splats_human_ids[gs_ids]
                # propagate the human ids to the other views
                gs_matching_graph = {}
                for human_ids in np.unique(gt_mask_split[cam_idx]):
                    if human_ids == 0:
                        continue
                    human_mask = gt_mask_split[cam_idx] == human_ids
                    values, counts = np.unique(propagated_human_id_mask[human_mask], return_counts=True)
                    counts = counts[values!=0]
                    values = values[values!=0]
                    for v, c in zip(values, counts):
                        if gs_matching_graph.get(v) is None:
                            gs_matching_graph[v] = {}
                        gs_matching_graph[v][human_ids] = c
                '''
                    gs_matching_graph:
                        key: gs_idx
                        value: {human_idx: count}   
                    
                    human_matching_graph:
                        key: human_idx
                        value: gs_idx
                '''
                human_matching_graph = {}
                for gs_idx, human_idx in gs_matching_graph.items():
                    max_hm_idx = max(human_idx, key=human_idx.get)
                    if human_matching_graph.get(max_hm_idx) is None:
                        human_matching_graph[max_hm_idx] = gs_idx
                    else:
                        if human_idx[max_hm_idx] > gs_matching_graph[human_matching_graph[max_hm_idx]][max_hm_idx]:
                            human_matching_graph[max_hm_idx] = gs_idx
                        
                for human_ids in np.unique(gt_mask_split[cam_idx]):
                    if human_ids == 0:
                        continue
                    if human_matching_graph.get(human_ids) is None:
                        cur_human_id += 1
                        human_matching_graph[human_ids] = cur_human_id
                
                for human_ids in np.unique(gt_mask_split[cam_idx]):
                    if human_ids == 0:
                        continue
                    human_mask = gt_mask_split[cam_idx] == human_ids
                    propagated_human_id_mask[human_mask] = human_matching_graph[human_ids]
                propagated_human_id_mask[gt_mask_comb_b_np[cam_idx]==0] = 0
                propagated_human_id_masks.append(propagated_human_id_mask)
                gs_human_ids = propagated_human_id_mask[pixel_uv[:, 1], pixel_uv[:, 0]]
                gs_ids = gs_ids[gs_human_ids!=0]
                gs_human_ids = gs_human_ids[gs_human_ids!=0]
                unmatch_gs_mask = splats_human_ids[gs_ids] == 0
                splats_human_ids[gs_ids[unmatch_gs_mask]] = gs_human_ids[unmatch_gs_mask]
        
              
        random_colors = np.random.random((cur_human_id, 3)) * 255
        # visualize 7 figures
        fig, axes = plt.subplots(2, 7, figsize=(40, 10))
        axes = axes.flatten()
        for cam_idx in range(len(data['novel_view'])):
            image = data['nonnorm_image'].cpu().numpy()[0, cam_idx].transpose(1, 2, 0).astype(np.uint8)
            # add mask to the image
            human_id_mask = propagated_human_id_masks[cam_idx]
            mask_color = np.zeros_like(image)#*255
            for human_id in np.unique(human_id_mask):
                if human_id == 0:
                    continue
                mask_color[human_id_mask == human_id] = random_colors[human_id - 1]
            image = cv2.addWeighted(image, 1.0, mask_color, 0.5, 0)
            image_m = mask_color 
            
            axes[cam_idx].imshow(image)  
            axes[cam_idx].set_title(f'Camera {cam_idx}')
            axes[cam_idx].axis('off')
            axes[cam_idx+7].imshow(image_m)  
            axes[cam_idx+7].set_title(f'Camera {cam_idx}')
            axes[cam_idx+7].axis('off')
            if save_frame is not None:
                cv2.imwrite(os.path.join(save_frame, f'{frame_idx}_{cam_idx}.png'), image[..., ::-1])
                cv2.imwrite(os.path.join(save_frame, f'{frame_idx}_{cam_idx}_m.png'), image_m[..., ::-1])
                
        # tightly fit layout
        plt.subplots_adjust(left=0, right=1, top=0.95, bottom=0.05, wspace=0.02, hspace=0.02)
        # plt.show()
        plt.savefig(os.path.join(save_folder, 'usp_cluster2D.png'), bbox_inches='tight', pad_inches=0, dpi=300)
        
        superpixel_data = sp_dataset[data_idx]
        centroids = superpixel_data['centroids']
        renders, _, _ = rasterize_splats(splats, w2cs, Ks, width, height)
        depths = renders[..., 3:4]
        all_pts3d_cnt_xyz = []
        all_pts3d_cnt_rgb = []
        all_pts3d_cls_idx = []
        
        num_cam = len(data['novel_view'])
        for cam_idx in range(num_cam):
            cnts = centroids[cam_idx]
            pts3d_cnt_idx = propagated_human_id_masks[cam_idx][cnts[:, 0].astype(np.int32), cnts[:, 1].astype(np.int32)] - 1
            pts3d_cnt_rgb = random_colors[pts3d_cnt_idx]
            
            depth_cnt = depths[cam_idx, cnts[:, 0].astype(np.int32), cnts[:, 1].astype(np.int32)].squeeze(-1).cpu().numpy()
            K = Ks[cam_idx].cpu().numpy()
            c2w = c2ws[cam_idx].cpu().numpy()
            x_cnt = (cnts[:, 1] - K[0, 2]) / K[0, 0] * depth_cnt 
            y_cnt = (cnts[:, 0] - K[1, 2]) / K[1, 1] * depth_cnt
            pts3d_cnt = np.stack([x_cnt, y_cnt, depth_cnt], axis=-1)
            pts3d_cnt = c2w[:3, :3] @ pts3d_cnt.T + c2w[:3, 3:4]
            all_pts3d_cnt_xyz.append(pts3d_cnt.T)   
            all_pts3d_cnt_rgb.append(pts3d_cnt_rgb)
            all_pts3d_cls_idx.append(pts3d_cnt_idx)
            
        all_pts3d_cnt_xyz = np.concatenate(all_pts3d_cnt_xyz, axis=0)
        all_pts3d_cnt_rgb = np.concatenate(all_pts3d_cnt_rgb, axis=0)
        all_pts3d_cls_idx = np.concatenate(all_pts3d_cls_idx, axis=0)
        
        pts3d_cnt_index = np.arange(all_pts3d_cnt_xyz.shape[0])
        pts3d_cnt_index_del = []
        for cam_idz in range(num_cam):
            
            pts3d_cnt_index_del.append(bg_filter_fn(w2c=w2cs[cam_idz].cpu().numpy(),
                        K=Ks[cam_idz].cpu().numpy(),
                        pts3d_world=all_pts3d_cnt_xyz.T,
                        pts3d_index=pts3d_cnt_index, 
                        masks_comb=gt_mask_comb_b[cam_idz].squeeze(-1).cpu().numpy(),
                        H=H, W=W))

        
        pts3d_cnt_index_del = np.concatenate(pts3d_cnt_index_del, axis=0)
        pts3d_cnt_index_del = np.unique(pts3d_cnt_index_del)
        pts3d_cnt_index_keep1 = np.setdiff1d(pts3d_cnt_index, pts3d_cnt_index_del)
        all_pts3d_cnt_xyz = all_pts3d_cnt_xyz[pts3d_cnt_index_keep1]
        all_pts3d_cnt_rgb = all_pts3d_cnt_rgb[pts3d_cnt_index_keep1]
        all_pts3d_cls_idx = all_pts3d_cls_idx[pts3d_cnt_index_keep1]
        
        t1 = time.time()
        print(f'Finish clustering and filtering in {t1-t0:.2f}s')
        
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(all_pts3d_cnt_xyz)
        pcd.colors = o3d.utility.Vector3dVector(all_pts3d_cnt_rgb / 255.0)
        o3d.io.write_point_cloud(os.path.join(save_folder, 'usp_cluster.ply'), pcd)
        # save the human id , xyz and rgb to the npy file
        np.savez(os.path.join(save_folder, 'usp_cluster.npz'), xyz=all_pts3d_cnt_xyz, rgb=all_pts3d_cnt_rgb, cls=all_pts3d_cls_idx)
        print('Finish clustering and saved in ', os.path.join(save_folder, 'usp_cluster.npz'))
        
        pcd = o3d.geometry.PointCloud()
        gs_mean = splats['means'].cpu().numpy()
        gs_mean = gs_mean[splats_human_ids>0]
        splats_human_ids = splats_human_ids[splats_human_ids>0]-1
        gs_rgb = random_colors[splats_human_ids]
        pcd.points = o3d.utility.Vector3dVector(gs_mean)
        pcd.colors = o3d.utility.Vector3dVector(gs_rgb / 255.0)
        o3d.io.write_point_cloud(os.path.join(save_folder, 'gs_cluster.ply'), pcd)
        print('Finish clustering and saved in ', os.path.join(save_folder, 'gs_cluster.ply'))
        
    
        

def bevMap2StandingPoint(bevmap, grid_scale=4):
    x, y = np.where(bevmap == 1)
    x = x * grid_scale
    y = y * grid_scale
    grid_xyz = np.stack([x, y, np.zeros_like(x)], axis=1)
    coord_xyz = np.zeros_like(grid_xyz)
    coord_xyz[:, 0] = -300 + 2.5 * grid_xyz[:, 0]
    coord_xyz[:, 1] = -900 + 2.5 * grid_xyz[:, 1]
    coord_xyz[:, 2] =   0  + 2.5 * grid_xyz[:, 2]
    return coord_xyz

def standingPoint2BEV(coord_xyz, grid_scale=4):
    x = (coord_xyz[:, 0] + 300) / 2.5
    y = (coord_xyz[:, 1] + 900) / 2.5
    x = np.clip(x, 0, 224)
    y = np.clip(y, 0, 224)
    x = x // grid_scale
    y = y // grid_scale
    bevmap = np.zeros((224, 224))
    bevmap[x.astype(np.int32), y.astype(np.int32)] = 1
    return bevmap

def localizeGS(args, vis_gs=False):
    
    print(args)
    data_root = args.root
    start_with = args.start_with # start to process from `start_with` th frame
    end_with = args.end_with #-1 # end to process at `end_with` th frame
    print(f'The start_with and end_with are: {start_with} and {end_with}')
    
    wildtrack_detect_data = frameDataset(WildtrackDetection('/home/jiahao/Downloads/data/Wildtrack'))
    
    wildtrack_data = Wildtrack(data_root, mask_type='people', mask_label_type='split', start_with=args.start_with, end_with=args.end_with)
    wildtrack_dataloader = DataLoader(wildtrack_data, batch_size=1, pin_memory=True, num_workers=4, drop_last=True)
    for data_idx, data in enumerate(wildtrack_dataloader):
        t0 = time.time()
        frame_idx = data['frame_idx'][0]
        
        detect_data_idx = int(frame_idx) // 5
        detect_data = wildtrack_detect_data[detect_data_idx]
        map_gt = detect_data[1]
        assert detect_data[3] == int(frame_idx), f'Frame index does not match {detect_data[3]} != {frame_idx}'
        coord_xyz = bevMap2StandingPoint(map_gt.squeeze(0).cpu().numpy())
        
        # H, W = data['H'].item(), data['W'].item()
        save_folder = os.path.join(data_root, 'depths', frame_idx, 'gs_optimize')
        clustergs_path = os.path.join(save_folder, 'usp_cluster.npz')   
        clustergs = np.load(clustergs_path)
        clustergs_xyz = clustergs['xyz']
        clustergs_rgb = clustergs['rgb']
        clustergs_cls = clustergs['cls']
        
        cluster_centers = []
        cluster_confidence = []
        for cls_idx in np.unique(clustergs_cls):
            gs_xyz = clustergs_xyz[clustergs_cls == cls_idx]
            if len(gs_xyz) < args.min_gs_threshold:
                continue
            # use the median xyz to represent the cluster's center
            # cluster_center = np.median(gs_xyz, axis=0)
            dbscan = DBSCAN(eps=300, min_samples=5)
            cluster_center_id = dbscan.fit_predict(gs_xyz)
            # if len(np.unique(cluster_center_id)) > 2:
            #     print(f'Cluster {cls_idx} has {len(np.unique(cluster_center_id))} centers')
            cluster_center = []
            for cluster_id in np.unique(cluster_center_id):
                if cluster_id == -1:
                    continue
                cls_cnts = np.median(gs_xyz[cluster_center_id == cluster_id], axis=0)
                cluster_confidence.append(len(gs_xyz[cluster_center_id == cluster_id]))
                cls_cnts[-1] = 0
                cluster_center.append(cls_cnts)
            if len(cluster_center) == 0:
                continue
            cluster_centers.append(np.stack(cluster_center))
        cluster_centers = np.concatenate(cluster_centers, axis=0)
        cluster_confidence = np.array(cluster_confidence)
        t1 = time.time()
        print(f'Processing frame {frame_idx} Inference time: {t1 - t0:.2f}s')
        np.savez( os.path.join(save_folder, 'usp_cluster_center.npz'), 
                 cluster_centers=cluster_centers, cluster_confidence=cluster_confidence,
                 clustergs_xyz=clustergs_xyz, clustergs_cls=clustergs_cls, 
                 clustergs_rgb=clustergs_rgb, gt_coord_xyz=coord_xyz)
        print('Finish clustering and saved in ', os.path.join(save_folder, 'usp_cluster_center.npz'))
        if vis_gs:
            meshes = []
            cluster_gs_pcd = o3d.geometry.PointCloud()
            cluster_gs_pcd.points = o3d.utility.Vector3dVector(clustergs_xyz)
            cluster_gs_pcd.colors = o3d.utility.Vector3dVector(clustergs_rgb/255.0)
            meshes.append(cluster_gs_pcd)

            for cnt in cluster_centers:
                cnt_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=10)
                cnt_sphere.translate(cnt)
                cnt_sphere.paint_uniform_color([1, 0, 0])
                meshes.append(cnt_sphere)

            for gt in coord_xyz:
                gt_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=10)
                gt_sphere.translate(gt)
                gt_sphere.paint_uniform_color([0, 1, 0])
                meshes.append(gt_sphere)
            o3d.visualization.draw_geometries(meshes)

def nms(heatmap, kernel=3, numThresh=0):
    local_max = ndi.maximum_filter(heatmap, size=kernel, mode='constant')
    nms_heatmap = np.where(heatmap == local_max, heatmap, 0)
    nms_heatmap = np.where(nms_heatmap > numThresh, 1, 0)
    return nms_heatmap  


def standingPoint2BEV(coord_xyz, confidences=None, grid_scale=4, x_range=(0, 480), y_range=(0, 1440)):
    '''
        x : [0, 480]  -> [0, 120]
        y : [0, 1440] -> [0, 360]
    '''
    x = (coord_xyz[:, 0] + 300) / 2.5
    y = (coord_xyz[:, 1] + 900) / 2.5
    z = coord_xyz[:, 2] / 2.5
    mask = (x >= x_range[0]) & (x < x_range[1]) & (y >= y_range[0]) & (y < y_range[1])
    x = x[mask]
    y = y[mask]
    z = z[mask]
    x = x // grid_scale
    y = y // grid_scale
    bevmap = np.zeros((x_range[1]//grid_scale, y_range[1]//grid_scale))
    if confidences is not None:
        bevmap[x.astype(np.int32), y.astype(np.int32)] = confidences[mask]
    else:
        bevmap[x.astype(np.int32), y.astype(np.int32)] = 1
    return bevmap

def construct_location(batch):
    xy = np.where(batch > 0)
    location = np.zeros((len(xy[0]), 3))
    location[:, 0] = xy[0]
    location[:, 1] = xy[1]
    return location

class FormatPRData():
    def __init__(self, save_dir) -> None:
        self.data = None
        self.save_dir = save_dir

    def add_item(self, batch, id):
        location = construct_location(batch)
        if self.data is None:
            self.data = np.concatenate([ np.ones((location.shape[0], 1))*id,  location], axis=1)
        else:
            tmp = np.concatenate([ np.ones((location.shape[0], 1))*id,  location], axis=1)
            self.data = np.concatenate([self.data, tmp], axis=0)
    def save(self):
        if not os.path.exists(os.path.dirname(self.save_dir)):
            os.mkdir(os.path.dirname(self.save_dir))
        np.savetxt(self.save_dir, self.data)
    
    def exist(self):
        return os.path.exists(self.save_dir)
    
def post_processGS(args):
    data_root = args.root
    # evaluate 40 frames following the existing evaluation
    evaluate_start_with = args.start_with
    evaluate_end_with = args.end_with
    wildtrack_data = Wildtrack(data_root, mask_type='people', mask_label_type='split', start_with=evaluate_start_with, end_with=evaluate_end_with)
    wildtrack_dataloader = DataLoader(wildtrack_data, batch_size=1, pin_memory=True, num_workers=4, drop_last=True)
    
    PR_pred = FormatPRData(args.pr_dir_pred)
    PR_gt = FormatPRData(args.pr_dir_gt)
    
    for data_idx, data in enumerate(wildtrack_dataloader):
        frame_idx = data['frame_idx'][0]
        save_folder = os.path.join(data_root, 'depths', frame_idx, 'gs_optimize')
        clustergs_path = os.path.join(save_folder, 'usp_cluster_center.npz') 
        clusters = np.load(clustergs_path)
        cluster_centers = clusters['cluster_centers']
        cluster_confidence = clusters['cluster_confidence']
        gt_coord_xyz = clusters['gt_coord_xyz']
        
        gt_map = standingPoint2BEV(gt_coord_xyz)

        pred_map = standingPoint2BEV(cluster_centers, cluster_confidence)
        pred_map = nms(pred_map, kernel=10, numThresh=15)
 
        # plt.figure(figsize=(10, 5))
        # plt.subplot(121)
        # plt.imshow(gt_map)
        # plt.title('Ground Truth')
        # plt.subplot(122)
        # plt.imshow(pred_map)
        # plt.title('Predicted')
        # plt.savefig('temp.png')

        PR_gt.add_item(gt_map, data_idx)
        PR_pred.add_item(pred_map, data_idx)
        
    PR_pred.save()
    PR_gt.save()
    recall, precision, moda, modp = evaluate_rcll_prec_moda_modp(args.pr_dir_pred, args.pr_dir_gt)
    print(f'\nEvaluation: MODA {moda:.1f}, MODP {modp:.1f}, prec {precision:.1f}, rcll {recall:.1f}')
    
    
if __name__ == '__main__':
    args_src = [
                '--root', '/home/jiahao/Downloads/data/wildtrack_data_gt',
                '--round', '1_2',
                '--n-segments', '30',
                '--start-with', '0',
                '--end-with', '10',
                '--min_gs_threshold' , '10'
            ]
    parser = ArgumentParser("Per scene training using Gaussian Splatting", add_help=True)
    parser.add_argument("--root", type=str, required=True, help="path to image folder")
    parser.add_argument("--round", type=str, required=True, help="round name")
    parser.add_argument("--start-with", type=int, default=0, help="the index of the first image to start with")
    parser.add_argument("--end-with", type=int, default=-1, help="the index of the last image to end with")
    parser.add_argument("--n-segments", type=int, default=30, help="the number of superpixels for each person")
    parser.add_argument("--min_gs_threshold", type=int, default=10, help="The minimum number of gaussian splats in a cluster for each human")
    parser.add_argument('--pr_dir_pred', type=str, 
                        default=r'output/exp_unsup/pr_dir_pred.txt')
    parser.add_argument('--pr_dir_gt', type=str, 
                        default=r'output/exp_unsup/pr_dir_gt.txt')
    args = parser.parse_args(args_src) # FOR DEBUG
    # args = parser.parse_args()
    
    # label matching
    clusterGS(args, save_frame='CVPR2025/clustering/figures')
    
    # clustering and localization
    localizeGS(args)

    # post-processing GS to get the final prediction
    post_processGS(args)