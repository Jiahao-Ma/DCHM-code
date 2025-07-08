import os
import cv2
import torch
import numpy as np
from copy import copy
import open3d as o3d
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from dataset.wildtrack import WildtrackDetection, frameDataset, Wildtrack, get_imagecoord_from_worldcoord
from ProteusLib import SuperPixelGaussian, render
from collections import defaultdict
def show_mask(image, mask_idx, mask, color=None, alpha=0.5):
    mask = np.where(mask == mask_idx, np.ones_like(mask), np.zeros_like(mask))
    if color is None:
        color = np.random.random(3) * 255
    mask = mask[..., None]
    # add color to the mask on the image
    if image.shape[-1] == 3:
        image = np.concatenate([image, np.ones_like(image[..., :1]) * 255], axis=-1)
    for i in range(3):  
        image[..., i] = image[..., i] * (1 - mask[..., 0] * alpha) + color[i] * mask[..., 0] * alpha

    return image, color

def multiview_check(ckpt_path='/home/jiahao/Downloads/data/wildtrack_data/depths/00000000/gs_1_1.ply'):
    import torch
    from torch.utils.data import DataLoader
    from ProteusLib import SuperPixelGaussian, render
    bg_filter = True
    depth_filter = True
    num_cam = 7
    depth_offset_threshold = 50
    round_name = '1_1'
    data_root = '/home/jiahao/Downloads/data/wildtrack_data'
    wildtrack_data = Wildtrack(data_root, mask_type='people', mask_label_type='split', start_with=0, end_with=-1)
    wildtrack_dataloader = DataLoader(wildtrack_data, batch_size=1, pin_memory=True, num_workers=4, drop_last=True)
    for idx, data in enumerate(wildtrack_dataloader):
        break
    masks_split = data['mask'].squeeze(0)
    
    masks_split_unqiue = np.zeros_like(masks_split)
    matching_graph = {}
    cur_match_idx = 1
    for i, mask in enumerate(masks_split):
        mask_unique = np.unique(mask.cpu().numpy())
        for m in mask_unique:
            if m == 0:
                continue
            masks_split_unqiue[i][m == mask] = cur_match_idx
            matching_graph[str(i)+':'+str(cur_match_idx)] = {}
            cur_match_idx += 1
    
    masks_comb = torch.where(masks_split>0, torch.ones_like(masks_split), torch.zeros_like(masks_split)).cpu().numpy()
    masks_comb = masks_comb.astype(np.bool_)
    c2ws = torch.cat([cam['c2w']for cam in  data['novel_view']], dim=0)
    w2cs = torch.cat([cam['w2c']for cam in  data['novel_view']], dim=0)
    Ks = torch.cat([cam['K']for cam in  data['novel_view']], dim=0)
    H, W = data['H'].item(), data['W'].item()
    
    images = data['nonnorm_image'].squeeze(0).cpu().numpy()
    images = (np.transpose(images, (0, 2, 3, 1))).astype(np.uint8)
    
    model = SuperPixelGaussian(None, max_sh_degree=3, required_grad=False)
    model.load_from_ckpt(ckpt_path)
    with torch.no_grad():   
        pred_imgs, pred_depths, pred_alphas, _ = render(model, data['novel_view'])
    pred_depths = pred_depths.detach().cpu().numpy().squeeze(-1)
    pred_imgs = pred_imgs.detach().cpu().numpy()
    
    all_pts_xyz = []
    all_pts_rgb = []
    all_uv_mask = []
    for cam_idx, (depth, img) in enumerate(zip(pred_depths, pred_imgs)):
         # diffuse depth 
        u = np.arange(0, W)
        v = np.arange(0, H)
        u, v = np.meshgrid(u, v)
        uv = np.stack([v, u], axis=-1).reshape(-1, 2)[masks_comb[cam_idx].reshape(-1)]
      
        selected_depth = depth[uv[:, 0].astype(np.int32), uv[:, 1].astype(np.int32)]
        pts3d_rgb = img[uv[:, 0].astype(np.int32), uv[:, 1].astype(np.int32)]
        K = Ks[cam_idx].cpu().numpy()
        x = (uv[:, 1] - K[0, 2]) / K[0, 0] 
        y = (uv[:, 0] - K[1, 2]) / K[1, 1]
        x = x.reshape(-1, 1) * selected_depth.reshape(-1, 1)
        y = y.reshape(-1, 1) * selected_depth.reshape(-1, 1)
        z = selected_depth.reshape(-1, 1)
        pts3d = np.concatenate([x, y, z], axis=-1) # (N, 3)
        c2w = c2ws[cam_idx].cpu().numpy()
        pts3d_world = c2w[:3, :3] @ pts3d.reshape(-1, 3).T + c2w[:3, 3:4]

        final_uv_mask = uv.copy()
        pts3d_index_keep1 = None
        pts3d_index_keep2 = None
        if bg_filter:
            pts3d_index = np.arange(pts3d_world.shape[1])
            pts3d_index_del = []
            for cam_idz in range(num_cam):
                w2c = w2cs[cam_idz].cpu().numpy()
                K = Ks[cam_idz].cpu().numpy()
                pts3d_cam = w2c[:3, :3] @ pts3d_world + w2c[:3, 3:4]
                pts3d_scrn = K @ pts3d_cam
                pts3d_scrn = pts3d_scrn[:2, :] / pts3d_scrn[2:3, :]
                mask1 = (pts3d_scrn[0, :] > 0) & (pts3d_scrn[0, :] < W) & (pts3d_scrn[1, :] > 0) & (pts3d_scrn[1, :] < H)
                pts3d_scrn = pts3d_scrn[:, mask1]
                mask2 = masks_comb[cam_idz]
                mask2 = mask2[pts3d_scrn[1].astype(np.int32), pts3d_scrn[0].astype(np.int32)]
                pts3d_index_del.append(pts3d_index[mask1][~mask2])
            pts3d_index_del = np.concatenate(pts3d_index_del, axis=0)
            pts3d_index_del = np.unique(pts3d_index_del)
            pts3d_index_keep1 = np.setdiff1d(pts3d_index, pts3d_index_del)
            pts3d_world = pts3d_world[:, pts3d_index_keep1]
            pts3d_rgb = pts3d_rgb[pts3d_index_keep1]
            final_uv_mask = final_uv_mask[pts3d_index_keep1]
        if depth_filter:
            pts3d_index = np.arange(pts3d_world.shape[1])
            pts3d_index_keep = []
            for cam_idy in range(num_cam):
                if cam_idy == cam_idx:
                    continue
                w2c = w2cs[cam_idy].cpu().numpy()
                K = Ks[cam_idy].cpu().numpy()
                pts3d_cam = w2c[:3, :3] @ pts3d_world + w2c[:3, 3:4]
                render_depth = pts3d_cam[-1, :]
                pts3d_scrn = K @ pts3d_cam
                pts3d_scrn = pts3d_scrn[:2, :] / pts3d_scrn[2:3, :]
                mask1 = (pts3d_scrn[0, :] > 0) & (pts3d_scrn[0, :] < W) & (pts3d_scrn[1, :] > 0) & (pts3d_scrn[1, :] < H)
                pts3d_scrn = pts3d_scrn[:, mask1]
                nearby_depth = pred_depths[cam_idy]
                nearby_depth = nearby_depth[pts3d_scrn[1].astype(np.int32), pts3d_scrn[0].astype(np.int32)]
                depth_dist = np.abs(nearby_depth - render_depth[mask1])
                mask2 = depth_dist < depth_offset_threshold
                pts3d_index_keep.append(pts3d_index[mask1][mask2])
            
            pts3d_index_keep2 = np.unique(np.concatenate(pts3d_index_keep, axis=0))
            pts3d_world = pts3d_world[:, pts3d_index_keep2]
            pts3d_rgb = pts3d_rgb[pts3d_index_keep2]
            
            final_uv_mask=final_uv_mask[pts3d_index_keep2]
        uv_mask_canvas = np.zeros((H, W), dtype=np.bool_)
        uv_mask_canvas[final_uv_mask[:, 0], final_uv_mask[:, 1]] = True
        
        # mask_norm = masks_split[cam_idx].cpu().numpy()
        # mask_norm = (mask_norm - mask_norm.min()) / (mask_norm.max() - mask_norm.min())
        fig, axes = plt.subplots(2, 4, figsize=(30, 20))
        axes = axes.flatten()
        image = images[cam_idx].copy()
        src_image = images[cam_idx].copy()
        valid_mask_idx = np.unique(masks_split[cam_idx][uv_mask_canvas])
        
        src_colors = []
        src_bbox = []
        src_masks = masks_split[cam_idx][final_uv_mask[:, 0], final_uv_mask[:, 1]]
        src_masks_unique = masks_split_unqiue[cam_idx][final_uv_mask[:, 0], final_uv_mask[:, 1]]
        for mask_idx in valid_mask_idx:
            
            human_region = np.where(mask_idx ==  masks_split[cam_idx])
            top_left = (np.min(human_region[1]), np.min(human_region[0]))  # (x_min, y_min)
            width = np.max(human_region[1]) - np.min(human_region[1])
            height = np.max(human_region[0]) - np.min(human_region[0])
            src_bbox.append((top_left[0], top_left[1], width, height))
            
            image, color = show_mask(image, mask_idx, masks_split[cam_idx])
            src_colors.append(color)
        axes[0].imshow(image)
        axes[0].axis('off') 
        axes[0].set_title(f'cam{cam_idx+1}')
        axidx = 1
        
        for cam_idz in range(num_cam):
            if cam_idx == cam_idz:
                continue
            w2c = w2cs[cam_idz].cpu().numpy()
            K = Ks[cam_idz].cpu().numpy()
            pts3d_cam = w2c[:3, :3] @ pts3d_world + w2c[:3, 3:4]
            render_depth = pts3d_cam[-1, :]
            pts3d_scrn = K @ pts3d_cam
            pts3d_scrn = pts3d_scrn[:2, :] / pts3d_scrn[2:3, :]
            mask1 = (pts3d_scrn[0, :] > 0) & (pts3d_scrn[0, :] < W) & (pts3d_scrn[1, :] > 0) & (pts3d_scrn[1, :] < H)
            pts3d_scrn = pts3d_scrn[:, mask1]
            
            nearby_depth = pred_depths[cam_idz]
            nearby_depth = nearby_depth[pts3d_scrn[1].astype(np.int32), pts3d_scrn[0].astype(np.int32)]
            depth_dist = np.abs(nearby_depth - render_depth[mask1])
            mask2 = depth_dist < depth_offset_threshold
            pts3d_scrn = pts3d_scrn[:, mask2]
            
            temp_image = images[cam_idz].copy()
            
            src_mask_index = src_masks[mask1][mask2]
            cor_mask = masks_split[cam_idz][pts3d_scrn[1].astype(np.int32), pts3d_scrn[0].astype(np.int32)]
            for m_idx in np.unique(cor_mask):
                unique, counts = np.unique(src_mask_index[cor_mask == m_idx], return_counts=True)
                most_unique = unique[np.argmax(counts)]   
                c = src_colors[np.where(valid_mask_idx == most_unique)[0].item()]
                temp_image, _ = show_mask(temp_image, m_idx, masks_split[cam_idz], color=c)

            src_mask_index_unique = src_masks_unique[mask1][mask2]
            cor_mask_unique = masks_split_unqiue[cam_idz][pts3d_scrn[1].astype(np.int32), pts3d_scrn[0].astype(np.int32)]
            for cur_idx in np.unique(cor_mask_unique):
                unique, counts = np.unique(src_mask_index_unique[cor_mask_unique == cur_idx], return_counts=True)
                src_idx = int(unique[np.argmax(counts)])
                # cur_idx: corresponding mask index in the current view
                # src_idx: corresponding mask index in the source view
                # one-to-one matching, if source view correspond to multiple masks, skip
                query_key = str(int(cam_idx)) + ":" + str(src_idx)
                if matching_graph[query_key].get(cam_idz) is not None:
                    # delete the previous matching
                    del matching_graph[query_key][cam_idz]
                    
                matching_graph[query_key][cam_idz] = int(cur_idx)
                
            axes[axidx].imshow(temp_image)
            # axes[axidx].scatter(pts3d_scrn[0], pts3d_scrn[1], c='r', s=10)
            axes[axidx].axis('off')
            axes[axidx].set_title(f'cam{cam_idz+1}')
            axidx+=1
        axes[-1].imshow(np.ones_like(image) * 255)
        axes[-1].axis('off')
        plt.subplots_adjust(wspace=0., hspace=0.) 
        plt.tight_layout(pad=0)
        # plt.show()
        plt.savefig(f'output/consistency_check/{cam_idx}.png', bbox_inches='tight', pad_inches=0, dpi=300)
        plt.close()
            
        all_uv_mask.append(uv_mask_canvas)
        all_pts_xyz.append(pts3d_world.T)
        all_pts_rgb.append(pts3d_rgb)
        
    all_uv_mask = np.stack(all_uv_mask, axis=0)
    all_pts_xyz = np.concatenate(all_pts_xyz, axis=0)
    all_pts_rgb = np.concatenate(all_pts_rgb, axis=0)
    
    # save matching graph
    np.save('output/consistency_check/matching_graph.npy', matching_graph)

def compare_matching_graph(file= 'output/consistency_check/matching_graph.npy'):
    matching_graph = np.load(file, allow_pickle=True).item()
    # for key, value in matching_graph.items():
    #     print(f'key: {key}')
    #     print(f'value: {value}')
    #     print('-------------------')
    data_root = '/home/jiahao/Downloads/data/wildtrack_data'
    wildtrack_data = Wildtrack(data_root, mask_type='people', mask_label_type='split', start_with=0, end_with=-1)
    data = wildtrack_data[0]
    masks_split = data['mask']
    
    masks_split_unqiue = np.zeros_like(masks_split)
    cur_match_idx = 1
    for i, mask in enumerate(masks_split):
        mask_unique = np.unique(mask)
        for m in mask_unique:
            if m == 0:
                continue
            masks_split_unqiue[i][m == mask] = cur_match_idx
            cur_match_idx += 1
    
    masks_comb = np.where(masks_split_unqiue>0, np.ones_like(masks_split), np.zeros_like(masks_split))

    images = data['nonnorm_image']
    images = (np.transpose(images, (0, 2, 3, 1))).astype(np.uint8)
    cur_match_len = 0
    for cam_idx in range(wildtrack_data.num_cam):
        fig, axes = plt.subplots(2, 4, figsize=(30, 20))
        axes = axes.flatten()
        image = images[cam_idx].copy()
        for mask_idx in np.unique(masks_split_unqiue[cam_idx]):
            query_key = str(int(cam_idx)) + ':' + str(int(mask_idx))
            image, color = show_mask(image, mask_idx, masks_split_unqiue[cam_idx])
        # matching = {0:{}, 1:{}, 2:{}, 3:{}, 4:{}, 5:{}, 6:{}}
        # for cam_idy in range(wildtrack_data.num_cam):
        #     image = images[cam_idy].copy()
        #     if cam_idx == cam_idy:
        #         for mask_idx in np.unique(masks_split_unqiue[cam_idy]):
        #             if mask_idx == 0:
        #                 continue
        #             query_key
        #             match_map = matching_graph[mask_idx]
        #             for k, v in match_map.items():
        #                 matching[k][v] = mask_idx   
        #             image, color = show_mask(image, mask_idx, masks_split_unqiue[cam_idy])
        #             src_colors.append(color)
        #         axes[cam_idy].imshow(image)
        #         axes[cam_idy].axis('off')
        #         axes[cam_idy].set_title(f'cam{cam_idy+1}')
                
        #     else:
        #         for mask_idx in np.unique(masks_split_unqiue[cam_idy]):
        #             if mask_idx == 0:
        #                 continue
        #             if matching[cam_idy].get(mask_idx) is None:
        #                 continue
        #             src_cor = int(matching[cam_idy][mask_idx])-1
        #             image, color = show_mask(image, mask_idx, masks_split_unqiue[cam_idy], color=src_colors[src_cor-cur_match_len])
        #             axes[cam_idy].imshow(image)
        #             axes[cam_idy].axis('off')
        #             axes[cam_idy].set_title(f'cam{cam_idy+1}')
        # cur_match_len += len(src_colors)
        # axes[-1].imshow(np.ones_like(image) * 255)
        # axes[-1].axis('off')
        # plt.subplots_adjust(wspace=0., hspace=0.) 
        # plt.tight_layout(pad=0)
        # # plt.show()
        # plt.savefig(f'output/consistency_check/{cam_idx}.png', bbox_inches='tight', pad_inches=0, dpi=300)
        # plt.close()

# if __name__ == "__main__":
#     multiview_check()
#     # compare_matching_graph()
#     compare_matching_graph_v1()