import os, sys;sys.path.append(os.getcwd())
sys.path.append(os.path.join(os.getcwd(), 'submodules/MvCHM/'))
import time
import torch, cv2
from torch import nn
import numpy as np
import matplotlib.pyplot as plt
from lib.utils.visual_utils import Process, Monitor
from lib.utils.tool_utils import to_numpy

from model.detector.crowdet_net import Crowdet, get_data
from model.detector.crowdet_config import crowdet_config
from model.detector.crowdet_lib import post_process

from model.refine.mspn_lib import mspn_preprocess, mspn_postprocess
from model.refine.mspn_config import mspn_cfg
from model.refine.mspn_net import MSPN

from model.ffe.depth_ffe import DepthFFE

from model.p2v.point2voxel import Point2Voxel

from model.pillar.pillar_vfe import PillarVFE
from model.pillar.pillar_scatter import PointPillarScatter

from model.m2b.bev_backbone import BaseBEVBackbone

from model.head.head_pred import Head

class MvCHM(nn.Module):
    def __init__(self, cfg, dataset, process:Process, device:torch.device, 
                 detector_ckpt = r'model\detector\checkpoint\rcnn_mxp.pth',
                 keypoint_ckpt = r'model\refine\checkpoint\mspn_mx.pth',
                 ):
        super().__init__()
        self.dataset = dataset
        self.device = device
        self.cfg = cfg

        # --- init detector ---  # 
        self.detector_config = crowdet_config
        self.detector = Crowdet().to(device=device)
        self.process = process

        # --- init keypoint --- # 
        self.keypoint = MSPN(mspn_cfg).to(device=device)

        # --- init DepthFFE module --- #
        self.ffe = DepthFFE(dataset, self.cfg, self.process).to(device=device)

        # --- init Point2Voxel module --- #
        self.p2v = Point2Voxel(self.cfg, self.process.reverse_image_feat).to(device=device)

        # ---- init PillarVFE module ---- #
        self.vfe = PillarVFE(self.cfg).to(device=device) # [n_pillars, 64]

        # ---- init PointPillarScatter module ---- #
        self.pps = PointPillarScatter(self.cfg).to(device=device) 

        # ---- init BaseBEVBackBone module ---- #
        self.bev = BaseBEVBackbone(self.cfg).to(device=device) 

        # ---- init Head module ---- #
        self.head = Head(self.cfg).to(device=device)

        # --- load model weight --- #
        if detector_ckpt is not None and keypoint_ckpt is not None:
            self._init_model_(detector_ckpt=detector_ckpt, keypoint_ckpt=keypoint_ckpt)
    
    def _init_model_(self, detector_ckpt = r'model\detector\checkpoint\rcnn_mxp.pth',
                           keypoint_ckpt = r'model\refine\checkpoint\mspn_mx.pth',
                           freeze_detector=True, freeze_keypoint=True): 
        
        assert os.path.exists(detector_ckpt), 'detection checkpoint path doesn\'t exist.' 
        self.detector.load_state_dict(torch.load(detector_ckpt, map_location='cuda:0')['state_dict']) # TODO: different cpu
        print('Detector module has loaded weight from %s.'%detector_ckpt)
        if freeze_detector:
            for param in self.detector.parameters():
                param.requires_grad = False

        assert os.path.exists(keypoint_ckpt), 'keypoint estimation checkpoint path doesn\'t exist.'
        self.keypoint.load_state_dict(torch.load(keypoint_ckpt, map_location='cuda:0')['model'])
        print('Keypoint estimation module has loaded weight from %s.'%keypoint_ckpt)
        if freeze_keypoint:
            for param in self.keypoint.parameters():
                param.requires_grad = False

    def _vis_bbox(self, batch_dict, vis_pred=False, vis_gt=False, save_root = r'visualization\mx_bbox', linewidth=1):
        kyps = None
        cam_num = len(batch_dict['images'])
        save_root = os.path.join(save_root, str(batch_dict['index']))
        if not os.path.exists(save_root):
            os.makedirs(save_root)
        for k in range(cam_num):
            img = to_numpy(batch_dict['images'][k]).astype(np.uint8)
            # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            plt.close()
            ax = plt.figure(figsize=(15, 8)).add_subplot(111)
            ax.imshow(img)
            ax.axis('off')
            box = batch_dict['pred_boxes'][k]
            if 'kyps_align' in batch_dict.keys():
                kyps = batch_dict['kyps_align'][k].squeeze(1)
            if vis_pred:
                for i in range(len(box)):
                    x1, y1, x2, y2 = box[i, :4]
                    if x1 == y1 == x2 == y2 == -1.:
                        continue
                    bbox = [int(x1), int(y1), int(x2), int(y2)]

                    one_box = np.array([max(bbox[0], 0), max(bbox[1], 0),
                            min(bbox[2], img.shape[1] - 1), min(bbox[3], img.shape[0] - 1)])
                    x1, y1, x2, y2 = one_box
                    # cv2.rectangle(np.ascontiguousarray(img), (x1, y1), (x2, y2), color=(0, 0, 255), thickness=4)
                    rect = plt.Rectangle((x1, y1), width=x2-x1, height=y2-y1, color='blue', linewidth=linewidth, fill=None)
                    ax.add_patch(rect)
                    if kyps is not None:
                        # cv2.circle(np.ascontiguousarray(img), center=(int(kyps[i][0]), int(kyps[i][1])), radius=6, color=(0,0,255), thickness=3)
                        kyps[i][0] = min(kyps[i][0], img.shape[1]-1)
                        kyps[i][1] = min(kyps[i][1], img.shape[0]-1)
                        ax.scatter(int(kyps[i][0]), int(kyps[i][1]), color='blue', linewidth=linewidth)
                        
               
                        
            if vis_gt:
                for gt_box in batch_dict['annot'][k]:
                    bbox = gt_box[:4]
                    bbox = np.array([max(bbox[0], 0), max(bbox[1], 0),
                            min(bbox[2], img.shape[1] - 1), min(bbox[3], img.shape[0] - 1)])
                    
                    rect = plt.Rectangle((bbox[0], bbox[1]), width=bbox[2]-bbox[0], height=bbox[3]-bbox[1], color='yellow', linewidth=linewidth, fill=None)
                    ax.add_patch(rect)
                    kyp = gt_box[4:]
                    kyp[-2] = min(kyp[-2], img.shape[1]-1)
                    kyp[-1] = min(kyp[-1], img.shape[0]-1)
                    ax.scatter(int(kyp[-2]), int(kyp[-1]), color='yellow', linewidth=linewidth)
            
            
            save_path = os.path.join(save_root, f"{k}.jpg")
            if os.path.exists(save_path):
                os.remove(save_path)
            # cv2.imwrite(os.path.join(save_path), img[..., ::-1])
            plt.savefig(save_path, bbox_inches='tight',dpi=300, pad_inches=0.0)
            plt.close()
        
        print(f'{save_root} has been saved!')

    def _vis_heatmap(self, batch_pred, gt, save_path='visualization\heatmap\wt_heatmap.jpg'):
        if not os.path.exists(os.path.dirname(save_path)):
            os.makedirs(os.path.dirname(save_path))
        heatmap = torch.sigmoid(batch_pred['heatmap']).squeeze(dim=0).squeeze(dim=-1)
        heatmap[heatmap<=0.7] = 0
        heatmap = heatmap.cpu().detach().numpy()
        heatmap = heatmap[:, :, np.newaxis].repeat(3, axis=-1) * 255
        plt.imshow(heatmap.astype(np.uint8))
        plt.axis('off')
        plt.title('prediction')
        plt.savefig(save_path, bbox_inches='tight', dpi=300, pad_inches=0.0)
        plt.close()
        gt_heatmap = gt['heatmap'][:, :, np.newaxis].repeat(3, axis=-1)
        gt_heatmap = (gt_heatmap* 255).astype(np.uint8)
        plt.imshow(gt_heatmap)
        plt.title('ground_truth')
        plt.axis('off')
        plt.savefig(save_path.split('.')[0]+'_gt.jpg', bbox_inches='tight', dpi=300, pad_inches=0.0)
        plt.close()
    
    def forward(self, data, save_pc=False, viz=False):
        batch_dict = {'annot':data['annot']}
        if 'index' in list(data.keys()):
            batch_dict['index'] = data['index']
        # --- 1. detection --- #
        self.detector.eval()
        post_pred_boxes = list()
        image_list = list()
        for img_path in data['image_paths']:
            image, resized_img, im_info = get_data(img_path, 800, 1400)
            resized_img = resized_img.to(device=self.device)
            pred_boxes = self.detector(resized_img, im_info)
            pred_boxes = post_process(pred_boxes, crowdet_config, im_info[0, 2])
            post_pred_boxes.append(pred_boxes)
            image_list.append(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        batch_dict['pred_boxes'] = post_pred_boxes # process the bbox 
        batch_dict['images'] = image_list

        # --- 2. keypoint estimation --- #
        # batch_dict['pred_boxes'] = data['annot']
        self.keypoint.eval()
        batch_dict = mspn_preprocess(batch_dict, self.cfg.MODEL.REFINE, self.device, self.process)
        batch_dict = self.keypoint(batch_dict)
        batch_dict = mspn_postprocess(batch_dict, mspn_cfg, \
            mspn_cfg.TEST.GAUSSIAN_KERNEL, mspn_cfg.TEST.SHIFT_RATIOS)
        
        if self.dataset.__name__ == 'Wildtrack':
            save_root = r'visualization\wt_bbox'
        else:
            save_root = r'visualization\mx_bbox'
        # self._vis_bbox(batch_dict, vis_pred=True, vis_gt=True, save_root=save_root)

        if not save_pc:
            batch_dict['images'] = self.process.normalize_image_array(batch_dict['images']).to(device=self.device) # (cams, 3, h, w)
        else:
            batch_dict['images'] = torch.stack([torch.from_numpy(img.transpose(2, 0, 1)) for img in batch_dict['images']], axis=0)
        batch_dict['image_size'] = batch_dict['images'].shape[-2:]
        
        # --- 3. regression --- #
        # --- 3.1 create frustum features --- #
        batch_dict = self.ffe(batch_dict, save_pc=save_pc, viz=viz)
        
        # ---- 3.2 convert points to pillar ---- #
        batch_dict = self.p2v(batch_dict)

        # ---- 3.3 create pillar features ---- #
        batch_dict = self.vfe(batch_dict)

        # ---- 3.4 create spatial features ---- #
        batch_dict = self.pps(batch_dict)

        # ---- 3.5 create compact spatial features ---- #
        batch_dict = self.bev(batch_dict)

        # ---- 3.6 prediction head ---- #
        batch_pred = self.head(batch_dict)
        
        # if self.dataset.__name__ == 'Wildtrack':
        #     save_root = f'visualization\\heatmap\\{data["index"]}.jpg'
        # else:
        #     save_root = r'visualization\heatmap\mx_heatmap.jpg'
        # self._vis_heatmap(batch_pred, data, save_path=save_root)

        return batch_dict, batch_pred



if __name__ == '__main__':
    import numpy as np
    from torchvision import transforms

    from lib.data.multiviewX import MultiviewX
    from lib.data.wildtrack import Wildtrack

    from torch.utils.data import DataLoader
    from lib.utils.config_utils import cfg, cfg_from_yaml_file
    from lib.utils.tool_utils import gen_scale
    from lib.data.dataloader import MultiviewDataset, collater, get_padded_value

    cfg_file = r'submodules/MvCHM/cfgs/MvDDE.yaml'
    cfg_from_yaml_file(cfg_file, cfg)

    new_w, new_h, old_w, old_h = cfg.DATA_CONFIG.NEW_WIDTH, cfg.DATA_CONFIG.NEW_HEIGHT, cfg.DATA_CONFIG.OLD_WIDTH, cfg.DATA_CONFIG.OLD_HEIGHT
    scale, scale_h, scale_w = gen_scale(new_w, new_h, old_w, old_h)
    pad_h, pad_w = get_padded_value(new_h, new_w)    

    process = Process(scale_h, scale_w, pad_h, pad_w, new_h, new_w, old_h, old_w)

    mean = torch.tensor(np.array([0.485, 0.456, 0.406]), dtype=torch.float32)
    std = torch.tensor(np.array([0.229, 0.224, 0.225]), dtype=torch.float32)

    dataname = 'Wildtrack'
    if dataname == 'MultiviewX':
        root = 'MultiviewX+_40'
    else:
        root = dataname
    path = '/home/jiahao/Downloads/data/{}'.format(dataname) # MultiviewX
    DATASET = {'MultiviewX': MultiviewX, 'Wildtrack': Wildtrack}

    dataset = MultiviewDataset( DATASET[dataname](root=path), set_name='val')
    dataloader_val = DataLoader( dataset, num_workers=1, batch_size=1, collate_fn=collater)

    device = torch.device('cuda:0')

    if dataname == 'MultiviewX':
        detector_ckpt = r'submodules/MvCHM/model/detector/checkpoint/rcnn_mxp.pth'
        # detector_ckpt = r'model\detector\checkpoint\rcnn_emd_refine.pth'
        keypoint_ckpt = r'submodules/MvCHM/model/refine/checkpoint/mspn_mx.pth'
    elif dataname == 'Wildtrack':
        detector_ckpt = r'submodules/MvCHM/model/detector/checkpoint/rcnn_wtp.pth'
        # detector_ckpt = r'model\detector\checkpoint\rcnn_emd_refine.pth'
        keypoint_ckpt = r'submodules/MvCHM/model/refine/checkpoint/mspn_wt.pth'
    mvchm = MvCHM(cfg, dataset, process, device, detector_ckpt=detector_ckpt, keypoint_ckpt=keypoint_ckpt)
    
    # ck_file = r'experiments\2022-10-23_19-53-52_wt\checkpoints\Epoch40_train_loss0.0608_val_loss0.0864.pth'
    # mvchm.load_state_dict(torch.load(ck_file, map_location=torch.device('cuda:0'))['model_state_dict'])
    # print(f'load checkpoint from {ck_file}.')
    
    monitor = Monitor()
    show = True
    for step, data in enumerate(dataloader_val):
       
        # batch_dict, batch_pred = mvchm(data)
        data['index'] = step
        mvchm(data, save_pc=True)
        