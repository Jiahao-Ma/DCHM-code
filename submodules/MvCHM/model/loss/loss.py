from multiprocessing import reduction
import torch
import torch.nn as nn
import torch.nn.functional as F
from lib.utils.tool_utils import to_tensor

def focal_loss(pred_heatmap, gt_heatmap, alpha=2., beta=4., eps=1e-5, reduction='mean', sigmoid=True):
    """
        Focal loss function for heatmap 
    """
    if sigmoid:
        pred = torch.sigmoid(pred_heatmap).clamp(eps, 1. - eps)
    else:
        pred = pred_heatmap.clamp(eps, 1. - eps)
    
    positive_mask = gt_heatmap == 1.
    negative_mask = ~positive_mask

    positive_num = positive_mask.sum()
    negative_num = negative_mask.sum()

    positive_loss = - (((1 - pred) ** alpha) * torch.log(pred)) * positive_mask.float()
    negative_loss = - (((1 - gt_heatmap) ** beta) * (pred ** alpha) * torch.log(1 - pred)) * negative_mask.float()
    if reduction == 'mean':
        positive_loss = torch.sum(positive_loss) / positive_num
        negative_loss = torch.sum(negative_loss) / negative_num
    elif reduction == 'sum':
        positive_loss = torch.sum(positive_loss) 
        negative_loss = torch.sum(negative_loss) 
    
    if positive_num == 0:
        return negative_loss
    elif negative_num == 0:
        return positive_loss
    else:
        return negative_loss + positive_loss


def compute_loss(batch_pred, batch_gt, loss_weight=[1., 1.]):
    """
        batch_gt:
            heatmap: (l, w) torch.tensor, the position of pedestrians on square
            offset: (l, w, 2) torch.tensor, the position offset on square grid
        batch_pred:
            pos_pred: (1, l, w, 1) torch.tensor, the prediction of position
            off_preds: (1, l, w, 2) torch.tensor, the prediction of position offset
    """
    # loss_offset_xy_fun = nn.SmoothL1Loss(reduction='none')
    # mask = torch.tensor(batch_gt['heatmap'] == 1.).to(dtype=torch.float32).unsqueeze(0).unsqueeze(-1)
    # batch_loss_offset_xy = loss_offset_xy_fun(torch.sigmoid(batch_pred['offset']), batch_gt['offset'].unsqueeze(0).to(batch_pred['offset'].device)) * mask
    
    batch_loss_heatmap = focal_loss(batch_pred['heatmap'], to_tensor(batch_gt['heatmap']).to(batch_pred['heatmap'].device).unsqueeze(0).unsqueeze(-1), reduction='mean')

    # batch_num_positive_samples = mask.sum()
    # batch_num_positive_samples = torch.maximum(batch_num_positive_samples, torch.ones_like(batch_num_positive_samples))

    # batch_loss_offset_xy /= batch_num_positive_samples
    # batch_loss_offset_xy = torch.sum(batch_loss_offset_xy)

    # loss = batch_loss_heatmap * loss_weight[0] + batch_loss_offset_xy * loss_weight[1]
    loss = batch_loss_heatmap
    
    batch_loss = { 'loss': loss.item(),
                   'loss_heatmap': batch_loss_heatmap.item() * loss_weight[1]
                 } 
    # batch_loss = { 'loss': loss.item(),
    #                'loss_offset': batch_loss_offset_xy.item() * loss_weight[0],
    #                'loss_heatmap': batch_loss_heatmap.item() * loss_weight[1]
    #              } 

    return loss, batch_loss

mseloss = nn.MSELoss(reduction='mean')

def compute_keypoint_loss(batch_dict):
    pred_maps = batch_dict['pred_kyp_heatmap']
    gt_maps = batch_dict['gt_kyp_heatmap']
    loss = 0
    for cam in range(len(pred_maps)):
        if len(gt_maps[cam]) == 0:
            continue  # No training samples
        pred_map = pred_maps[cam].squeeze(1)
        device = pred_map.device
        gt_map = gt_maps[cam].squeeze(1).to(device=device)
        # loss += focal_loss(pred_map, gt_map, reduction='mean', sigmoid=False)
        loss += mseloss(pred_map, gt_map)

    batch_loss = { 'loss': loss.item() } 
    return loss, batch_loss
