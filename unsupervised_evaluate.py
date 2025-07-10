import os
import numpy as np
from argparse import ArgumentParser
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from dataset.wildtrack import Wildtrack
from utils.evaluation.evaluate import evaluate_rcll_prec_moda_modp
from unsupervised_cluster import FormatPRData, standingPoint2BEV, nms

def Evaluate(args):
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
    # args = parser.parse_args(args_src) # FOR DEBUG
    args = parser.parse_args()
    
    # post-processing GS to get the final prediction
    Evaluate(args)