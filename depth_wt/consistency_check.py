import os, sys;sys.path.append(os.getcwd())
import cv2
import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
from dataset.wildtrack import WildtrackDetection, frameDataset, Wildtrack, get_imagecoord_from_worldcoord
import matplotlib.patches as patches
def generate_colors(n):
    colors = plt.cm.get_cmap('hsv', n)
    return np.array([colors(i)[:3] for i in range(n)])

def create_bounding_box(center, width=50, height=100, depth=50):
    # Create a bounding box with the given dimensions
    bbox = o3d.geometry.OrientedBoundingBox()
    bbox.extent = np.array([width, depth, height])
    bbox.center = center + np.array([0, 0, height / 2])
    return bbox

def find_the_GS_in_bbox(vis_canvas=False, 
                        pcd_path = r'output/superpixel_train/w_sparsity/epoch_999.ply',
                        vis_type='sparse',
                        highlight_within_bbox=False,
                        vis_ply_idx=True,
                        vis_terval=10
                        ):
    human_pcd = o3d.io.read_point_cloud(pcd_path)
    # full black color for the point cloud
    human_pcd.colors = o3d.utility.Vector3dVector(np.zeros_like(np.asarray(human_pcd.points)))
    point_cloud_idx_path = os.path.join(os.path.dirname(pcd_path), 'point_cloud_idx.npy')
    if os.path.exists(point_cloud_idx_path) and vis_ply_idx:
        point_cloud_idx = np.load(point_cloud_idx_path)
        point_cloud_idx = point_cloud_idx.astype(np.int32)
        point_cloud_max_idx = np.max(point_cloud_idx)
    else:
        point_cloud_idx = None
    if point_cloud_idx is not None:
        
        color_template = generate_colors(point_cloud_max_idx+1)
        new_colors = np.array([color_template[idx] for idx in point_cloud_idx]).reshape(-1, 3)
        human_pcd.colors = o3d.utility.Vector3dVector(new_colors)
        
        
    dataset = frameDataset(WildtrackDetection('/home/jiahao/Downloads/data/Wildtrack'))
    data = dataset[0]
    imgs, map_gt, imgs_gt, frame, _ = data 
    map_gt = map_gt.cpu().numpy()[0]   
    grid_scale = 4
    x, y = np.where(map_gt == 1)
    x = x * grid_scale
    y = y * grid_scale
    human_grid_xyz = np.stack([x, y, np.zeros_like(x)], axis=1)
    human_coord_xyz = np.zeros_like(human_grid_xyz)
    human_coord_xyz[:, 0] = -300 + 2.5 * human_grid_xyz[:, 0]
    human_coord_xyz[:, 1] = -900 + 2.5 * human_grid_xyz[:, 1]
    human_coord_xyz[:, 2] =   0  + 2.5 * human_grid_xyz[:, 2]
    
    corner_grid = np.array([[0, 0, 0], [0, 1440, 0], [480, 0, 0], [480, 1440, 0]]).reshape(-1, 3)
    corner_coord = np.zeros_like(corner_grid)
    corner_coord[:, 0] = -300 + 2.5 * corner_grid[:, 0]
    corner_coord[:, 1] = -900 + 2.5 * corner_grid[:, 1]
    corner_coord[:, 2] =   0  + 2.5 * corner_grid[:, 2]
    # np.save('CVPR2025/main_pipeline/map_gt.npy', map_gt)
    # np.save('CVPR2025/main_pipeline/human_coord_xyz.npy', human_coord_xyz)
    # np.save('CVPR2025/main_pipeline/corner_coord.npy', corner_coord)
    # Visualize the human coordinates and corner rectangle
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    
    witihin_indices = []
    cor_bbox_human_idx = []
    no_cor_bbox_human_idx = []
    # Create and add bounding boxes for each human coordinate
    human_standing_pc_colors = generate_colors(len(human_coord_xyz))
    for idx, coord in enumerate(human_coord_xyz):
        bbox = create_bounding_box(coord, height=200, width=100, depth=100)
        bbox.color = np.array([0, 1, 0])  # Green color for the bounding boxes
        vis.add_geometry(bbox)
        
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=10)
        sphere.translate(coord)
        sphere.paint_uniform_color(human_standing_pc_colors[idx])  # Red color for the spheres
        vis.add_geometry(sphere)
        
        # Highlight points inside the bounding box in blue
        indices = bbox.get_point_indices_within_bounding_box(human_pcd.points)
        # print(f'[{idx} bbox] has {len(indices)} points')
        if len(indices) > 0:
            cor_bbox_human_idx.append(idx)
        else:
            no_cor_bbox_human_idx.append(idx)
        if highlight_within_bbox:
            for i, within_coord in enumerate(np.asarray(human_pcd.points)[indices]):
                if vis_type == 'dense':
                    pass
                elif vis_type == 'sparse':
                    if i % vis_terval != 0:
                        continue
                within_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=10)
                within_sphere.translate(within_coord)
                within_sphere.paint_uniform_color([0, 0, 1])  # Blue color for the spheres
                vis.add_geometry(within_sphere)
            
        witihin_indices.append(indices)
    # Add the point cloud to the visualizer
    vis.add_geometry(human_pcd)
    
    # Create lines to form the rectangle using the corner coordinates
    lines = [[0, 1], [1, 3], [3, 2], [2, 0]]
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(corner_coord)
    line_set.lines = o3d.utility.Vector2iVector(lines)
    line_set.colors = o3d.utility.Vector3dVector([[1, 0, 0] for _ in range(len(lines))])  # Red color for the lines
    vis.add_geometry(line_set)
    # Update the point cloud with the new colors
    vis.update_geometry(human_pcd)
    
    if vis_canvas:
        vis.poll_events()
        vis.update_renderer()
        
        # Run the visualizer
        vis.run()
        vis.destroy_window()
    return witihin_indices, human_pcd, dataset, human_coord_xyz, human_standing_pc_colors, cor_bbox_human_idx, no_cor_bbox_human_idx

def get_worldcoord_from_worldgrid(worldgrid):
        # datasets default unit: centimeter & origin: (-300,-900)
        grid_x, grid_y = worldgrid
        coord_x = -300 + 2.5 * grid_x
        coord_y = -900 + 2.5 * grid_y
        return np.array([coord_x, coord_y])

def project(pts, intr, extr, cur_w = 1920, cur_h = 1080, tar_w = 480, tar_h = 270):
    pts_cam = extr[:3, :3] @ pts.T  + extr[:3, 3].reshape(3, 1)
    pts_cam = intr @ pts_cam 
    pts_cam = pts_cam[:2] / pts_cam[2:]
    pts_cam = pts_cam.T
    valid_mask = (pts_cam[:, 0] > 0) & (pts_cam[:, 0] < cur_w-1) & (pts_cam[:, 1] > 0) & (pts_cam[:, 1] < cur_h-1)
    pts_cam = pts_cam[valid_mask]
    pts_cam[:, 0] = pts_cam[:, 0] / cur_w * tar_w
    pts_cam[:, 1] = pts_cam[:, 1] / cur_h * tar_h
    return pts_cam, valid_mask
    
    
def find_the_GS_in_2DImage(pcd_path, image_path = r'/home/jiahao/Downloads/data/wildtrack_data/Image_subsets/00000000', vis_within_pc=False):
    
    
    witihin_indices, human_pcd, dataset3D, human_gt_xyz, human_standing_pc_colors, cor_bbox_human_idx, no_cor_bbox_human_idx =\
        find_the_GS_in_bbox(vis_canvas=False, pcd_path=pcd_path)
    human_idx_txt = np.arange(human_standing_pc_colors.shape[0])
    human_gt_xyz = np.array(human_gt_xyz)

    human_gs_xyz = np.array(human_pcd.points)
    human_gs_rgb = np.array(human_pcd.colors)
    
    human_within_gs_xyz = human_gs_xyz[np.concatenate(witihin_indices).astype(np.int32)]
    dataset3D = frameDataset(WildtrackDetection('/home/jiahao/Downloads/data/Wildtrack'))
    
    for i in range(7):
        gt_img = cv2.imread(os.path.join(image_path, f'cam{i+1}.png'))
        gt_img = np.array(gt_img)[..., ::-1]
        tar_h, tar_w = gt_img.shape[:2]
        human_gt_xyz_, gt_valid_mask = project(human_gt_xyz, dataset3D.base.intrinsic_matrices[i], dataset3D.base.extrinsic_matrices[i], tar_h=tar_h, tar_w=tar_w)
        human_gs_xyz_, gs_valid_mask = project(human_gs_xyz, dataset3D.base.intrinsic_matrices[i], dataset3D.base.extrinsic_matrices[i], tar_h=tar_h, tar_w=tar_w)
        
        
        plt.figure(figsize=(30, 30))
        plt.imshow(gt_img)
        # visualize the  img_coord using scatter
        plt.scatter(human_gt_xyz_[:, 0], human_gt_xyz_[:, 1], c=human_standing_pc_colors[gt_valid_mask, :], s=200, marker='s')
        for scn, txt in zip(human_gt_xyz_, human_idx_txt[gt_valid_mask]):
            if txt in cor_bbox_human_idx:
                plt.text(scn[0], scn[1], f'{txt}', fontsize=25, color='green', fontweight='bold')
            else:
                plt.text(scn[0], scn[1], f'{txt}', fontsize=25, color='red', fontweight='bold')

            
        if vis_within_pc:
            plt.scatter(human_gs_xyz_[:, 0], human_gs_xyz_[:, 1], c=human_gs_rgb[gs_valid_mask, :], s=30, alpha=0.8)
            human_within_gs_xyz_ = project(human_within_gs_xyz, dataset3D.base.intrinsic_matrices[i], dataset3D.base.extrinsic_matrices[i], tar_h=tar_h, tar_w=tar_w)
            plt.scatter(human_within_gs_xyz_[:, 0], human_within_gs_xyz_[:, 1], c='b', s=30)
        plt.axis('off')
        plt.show()
        # if os.path.exists(f'output/consistency_check') == False:
        #     os.makedirs(f'output/consistency_check')
        # plt.savefig(f'output/consistency_check/cam{i+1}.png', bbox_inches='tight', pad_inches=0, dpi=300)
        # plt.close()

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
    depth_offset_threshold = 100
    round_name = '1_1'
    data_root = '/home/jiahao/Downloads/data/wildtrack_data'
    wildtrack_data = Wildtrack(data_root, mask_type='people', mask_label_type='split', start_with=0, end_with=-1)
    wildtrack_dataloader = DataLoader(wildtrack_data, batch_size=1, pin_memory=True, num_workers=4, drop_last=True)
    for idx, data in enumerate(wildtrack_dataloader):
        break
    masks_split = data['mask'].squeeze(0)
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
            tar_image = images[cam_idz].copy()
            src_mask_index = src_masks[mask1][mask2]
            cor_mask = masks_split[cam_idz][pts3d_scrn[1].astype(np.int32), pts3d_scrn[0].astype(np.int32)]
            for m_idx in np.unique(cor_mask):
                unique, counts = np.unique(src_mask_index[cor_mask == m_idx], return_counts=True)
                most_unique = unique[np.argmax(counts)]   
                c = src_colors[np.where(valid_mask_idx == most_unique)[0].item()]
                temp_image, _ = show_mask(temp_image, m_idx, masks_split[cam_idz], color=c)
                
                #TODO: add XFeat matching here
                # human_region = np.where(m_idx ==  masks_split[cam_idz])
                # top_left = (np.min(human_region[1]), np.min(human_region[0]))  # (x_min, y_min)
                # width = np.max(human_region[1]) - np.min(human_region[1])
                # height = np.max(human_region[0]) - np.min(human_region[0])
                
                # src_b = src_bbox[np.where(valid_mask_idx == most_unique)[0].item()] # top_lef, width, height
                # src_img = src_image[src_b[1]:src_b[1]+src_b[3], src_b[0]:src_b[0]+src_b[2], :3]
                # tar_img = tar_image[top_left[1]:top_left[1]+height, top_left[0]:top_left[0]+width]
                # # pad the image that can be divided by 32
                # src_img = np.pad(src_img, ((0, 32-src_img.shape[0]%32), (0, 32-src_img.shape[1]%32), (0, 0)), mode='constant')
                # tar_img = np.pad(tar_img, ((0, 32-tar_img.shape[0]%32), (0, 32-tar_img.shape[1]%32), (0, 0)), mode='constant')
                # points1, points2 = match_features(src_img, tar_img, max_kpts=4096, min_cossim=0.9)
                # # Draw matches and show the result
                # matched_img = draw_matches(src_img, tar_img, points1, points2)
                # matched_img = cv2.resize(matched_img, (1280, 960))
                # cv2.imshow("Matched Features", matched_img[..., ::-1])
                # cv2.waitKey(0)
                # cv2.destroyAllWindows()
                
                # draw the bounding box
                # rect = patches.Rectangle(top_left, width, height, linewidth=2, edgecolor='red', facecolor='none')
                # axes[axidx].add_patch(rect)
            
            
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
    
    # pcd = o3d.geometry.PointCloud()
    # pcd.points = o3d.utility.Vector3dVector(all_pts_xyz)
    # pcd.colors = o3d.utility.Vector3dVector(all_pts_rgb / 255)
    # o3d.io.write_point_cloud(os.path.join(os.path.dirname(ckpt_path), f'diffuse_gs_{round_name}.ply'), pcd)
    
    # all_depth_masks = all_uv_mask
    # mask_img = np.concatenate(all_depth_masks, axis=1)
    # np.save(os.path.join(os.path.dirname(ckpt_path), f'depth_{round_name}.npy'), pred_depths)
    # np.save(os.path.join(os.path.dirname(ckpt_path), f'mask_{round_name}.npy'), all_depth_masks)
    # cv2.imwrite(os.path.join(os.path.dirname(ckpt_path), f'mask_{round_name}.png'), mask_img.astype(np.uint8) * 255)
    
def mask_selection(min_area = 500):
    mask_path = r'/home/jiahao/Downloads/data/wildtrack_data/masks/people/00000000'
    img_path = r'/home/jiahao/Downloads/data/wildtrack_data/Image_subsets/00000000'
    masks = []
    images = []
    for cam_idx in range(1, 8):
        masks.append(np.load(os.path.join(mask_path, f'cam{cam_idx}.npy')))
        images.append(cv2.imread(os.path.join(img_path, f'cam{cam_idx}.png')))
    fig, axes = plt.subplots(2, 4, figsize=(30, 20))
    axes = axes.flatten()
    for id, mask in enumerate(masks):
        for mask_id in np.unique(mask):
            if mask_id == 0:
                continue
            mask_area = np.sum(mask == mask_id)
            if mask_area < min_area:
                mask[mask == mask_id] = 0
        # draw the mask on the image
        mask = np.where(mask>0, np.ones_like(mask), np.zeros_like(mask))    
        
        mask_img = np.stack([mask, mask, mask], axis=-1)
        mask_img = mask_img * 255
        mask_img = mask_img.astype(np.uint8)
        mask_img = cv2.addWeighted(images[id], 0.5, mask_img, 0.5, 0)
        axes[id].imshow(mask_img[..., ::-1])
        axes[id].axis('off')
        axes[id].set_title(f'cam{id+1}')
    axes[-1].axis('off')    
    axes[-1].imshow(np.ones_like(mask_img) * 255)
    plt.tight_layout()
    plt.show()
    
if __name__ == "__main__":
    find_the_GS_in_bbox(vis_canvas=True, 
                        vis_type='sparse',
                        highlight_within_bbox=True,
                        vis_ply_idx=False,
                        vis_terval=50,
                        pcd_path =r'CVPR2025/point_cloud/point_cloud_ours.ply'
                        # pcd_path =r'/media/downloads/data/wildtrack_data/depths/00000000/diffuse_gs_1_2.ply'
                        )
    # find_the_GS_in_2DImage( 
    #     image_path = r'/home/jiahao/Downloads/data/wildtrack_data/Image_subsets/00000000',
    #     pcd_path =r'/home/jiahao/Downloads/data/wildtrack_data/depths/00000000/diffuse_gs_1_1.ply'
    #     )
    # multiview_check()
    # mask_selection(min_area=2000)
