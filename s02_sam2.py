import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import argparse
import sys

import numpy as np
import torch

from tqdm import tqdm
import matplotlib.patches as patches
from multiprocessing import Pool, cpu_count

sys.path.append(os.path.join(os.getcwd(), "submodules/Grounded-Segment-Anything"))
from grounded_sam_demo import *

def get_gt_bboxes(frame_idx, cam_id, data_dir=r'/media/downloads/data/Wildtrack', w=1920, h=1080):
    annotation_dir_out = os.path.join(data_dir, 'annotations_positions_all')
    json_files_out = os.path.join(annotation_dir_out, f'{frame_idx:08d}_all.json')    
    annotation_dir_int = os.path.join(data_dir, 'annotations_positions')
    json_files_int = os.path.join(annotation_dir_int, f'{frame_idx:08d}.json')    
    with open(json_files_out, 'r') as f:
        annotations_out = json.load(f)
    with open(json_files_int, 'r') as f:
        annotations_int = json.load(f)
    bbox_out = []
    bbox_int = []
    for outsider in annotations_out[cam_id]['outsiders']:
        xmin = outsider['xmin']
        ymin = outsider['ymin']
        xmax = min(outsider['xmax'], w)
        ymax = min(outsider['ymax'], h)
        bbox_out.append([xmin, ymin, xmax, ymax])
    for insider in annotations_int:
        bbox = insider['views'][cam_id]
        if bbox['xmax'] == -1 and bbox['xmin'] == -1:
            continue
        
        xmin = bbox['xmin']
        ymin = bbox['ymin']
        xmax = min(bbox['xmax'], w)
        ymax = min(bbox['ymax'], h)
        
        bbox_int.append([xmin, ymin, xmax, ymax])
    bbox_out = torch.tensor(bbox_out)
    bbox_int = torch.tensor(bbox_int)
    bboxes = torch.cat([bbox_out, bbox_int], dim=0)
    return bboxes

def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def show_box(box, ax, label):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2))
    ax.text(x0, y0, label)


def save_mask_jpg(mask_list, save_name):
    mask_img = torch.zeros(mask_list.shape[-2:])
    for idx, mask in enumerate(mask_list):
        mask_img[mask.cpu().numpy()[0] == True] = 1
    
    np.save(save_name.replace('.png', '.npy'), mask_img.cpu().numpy())

    masks = np.where(mask_list.squeeze(1).cpu().numpy())
    np.save(save_name.split('.')[0]+'_split' + '.npy', masks)
        
def save_mask_json(output_dir, save_name, mask_list, box_list, label_list, image):
    value = 0  # 0 for background
    cur_mask_id = 0
    mask_img = torch.zeros(mask_list.shape[-2:])
    for mask in mask_list:
        selected_pixels = mask_img[mask.cpu().numpy()[0] == True] 
        filled_region = len(selected_pixels[selected_pixels > 0])
        total_region = len(selected_pixels)
        if total_region == 0:
            continue
        if filled_region / total_region > 0.8:
            continue
        mask_img[mask.cpu().numpy()[0] == True] = value + cur_mask_id + 1
        cur_mask_id += 1
    
    # cv2.imwrite(os.path.join(output_dir, f'{save_name}.jpg'), mask_img.numpy().astype(np.uint8))
    np.save(os.path.join(output_dir, f'{save_name}.npy'), mask_img.cpu().numpy().astype(np.uint8))
    json_data = [{
        'value': value,
        'label': 'background'
    }]
    for label, box in zip(label_list, box_list):
        value += 1
        name, logit = label.split('(')
        logit = logit[:-1] # the last is ')'
        json_data.append({
            'value': value,
            'label': name,
            'logit': float(logit),
            'box': box.numpy().tolist(),
        })
    with open(os.path.join(output_dir, f'{save_name}.json'), 'w') as f:
        json.dump(json_data, f)

    # draw the mask for the image
    mask_img = mask_img.numpy()
    mask_img = np.stack([mask_img, mask_img, mask_img], axis=-1)
    mask_img = mask_img * 255
    mask_img = mask_img.astype(np.uint8)
    mask_img = cv2.addWeighted(image, 0.5, mask_img, 0.5, 0)
    cv2.imwrite(os.path.join(output_dir, f'{save_name}_mask.jpg'), cv2.cvtColor(mask_img, cv2.COLOR_RGB2BGR))
    
def main():
    target = 'ground'
    root = "/home/jiahao/Downloads/data/wildtrack_data"
    args=[  
            "--root", root,
            "--config", "submodules/Grounded-Segment-Anything/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py",
            "--grounded_checkpoint", "submodules/Grounded-Segment-Anything/groundingdino_swint_ogc.pth",
            "--sam_checkpoint", "submodules/Grounded-Segment-Anything/sam_vit_h_4b8939.pth", #[Default]
            # Download sam_hq_vit_h.pth from https://github.com/SysCV/sam-hq#model-checkpoints
            # "--sam_checkpoint", "submodules/Grounded-Segment-Anything/sam_hq_vit_h.pth", 
            # "--use_sam_hq",
            "--input_image", None,
            "--output_dir", os.path.join(root, f"masks/{target}"),
            "--box_threshold", "0.3",
            "--text_threshold", "0.25", 
            "--text_prompt", target,
            "--device", "cuda:0"
        ]
    parser = argparse.ArgumentParser("Grounded-Segment-Anything Demo", add_help=True)
    parser.add_argument("--root", type=str, required=True, help="path to image folder")
    parser.add_argument("--config", type=str, required=True, help="path to config file")
    parser.add_argument(
        "--grounded_checkpoint", type=str, required=True, help="path to checkpoint file"
    )
    parser.add_argument(
        "--sam_version", type=str, default="vit_h", required=False, help="SAM ViT version: vit_b / vit_l / vit_h"
    )
    parser.add_argument(
        "--sam_checkpoint", type=str, required=False, help="path to sam checkpoint file"
    )
    parser.add_argument(
        "--sam_hq_checkpoint", type=str, default=None, help="path to sam-hq checkpoint file"
    )
    parser.add_argument(
        "--use_sam_hq", action="store_true", help="using sam-hq for prediction"
    )
    parser.add_argument("--input_image", type=str, required=True, help="path to image file")
    parser.add_argument("--text_prompt", type=str, required=True, help="text prompt")
    parser.add_argument(
        "--output_dir", "-o", type=str, default="outputs", required=True, help="output directory"
    )

    parser.add_argument("--box_threshold", type=float, default=0.3, help="box threshold")
    parser.add_argument("--text_threshold", type=float, default=0.25, help="text threshold")

    parser.add_argument("--device", type=str, default="cpu", help="running on cpu only!, default=False")
    args = parser.parse_args(args) # FOR DEBUG
    # args = parser.parse_args()

    # cfg
    root = os.path.join(args.root, 'Image_subsets')
    config_file = args.config  # change the path of the model config file
    grounded_checkpoint = args.grounded_checkpoint  # change the path of the model
    sam_version = args.sam_version
    sam_checkpoint = args.sam_checkpoint
    sam_hq_checkpoint = args.sam_hq_checkpoint
    use_sam_hq = args.use_sam_hq
    # image_path = args.input_image
    text_prompt = args.text_prompt
    output_dir = args.output_dir
    box_threshold = args.box_threshold
    text_threshold = args.text_threshold
    device = args.device
    print('device:', device)

    # make dir
    os.makedirs(output_dir, exist_ok=True)
    # load model
    model = load_model(config_file, grounded_checkpoint, device=device)
    # initialize SAM
    if use_sam_hq:
        predictor = SamPredictor(sam_hq_model_registry[sam_version](checkpoint=sam_hq_checkpoint).to(device))
    else:
        predictor = SamPredictor(sam_model_registry[sam_version](checkpoint=sam_checkpoint).to(device))
        
    # root =r'/home/jiahao/Downloads/data/wildtrack/Image_subsets'
    print('root:', root)
    with torch.no_grad():
        for image_folder in sorted(os.listdir(root)):
            image_folder_path = os.path.join(root, image_folder)
            print('processing folder:', image_folder_path)
            save_folder = os.path.join(output_dir, image_folder)
            if not os.path.exists(save_folder):
                os.makedirs(save_folder, exist_ok=True)
            # if os.path.exists(os.path.join(save_folder, 'cam7.npy')):
            #     print(f'{save_folder} has been processed, skip')
            #     continue
            for image_name in tqdm(sorted(os.listdir(image_folder_path))):
                image_path = os.path.join(image_folder_path, image_name)
        
                # load image
                image_pil, image = load_image(image_path)

                # run grounding dino model
                boxes_filt, pred_phrases = get_grounding_output(
                    model, image, text_prompt, box_threshold, text_threshold, device=device
                )
                image = cv2.imread(image_path)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                
                if boxes_filt.size(0) > 0:
                    predictor.set_image(image)

                    size = image_pil.size
                    H, W = size[1], size[0]
                    for i in range(boxes_filt.size(0)):
                        boxes_filt[i] = boxes_filt[i] * torch.Tensor([W, H, W, H])
                        boxes_filt[i][:2] -= boxes_filt[i][2:] / 2
                        boxes_filt[i][2:] += boxes_filt[i][:2]

                    boxes_filt = boxes_filt.cpu()
                    
                    transformed_boxes = predictor.transform.apply_boxes_torch(boxes_filt, image.shape[:2]).to(device)
                    
                    masks, _, _ = predictor.predict_torch(
                        point_coords = None,
                        point_labels = None,
                        boxes = transformed_boxes.to(device),
                        multimask_output = False,
                    )
                    
    
                    # save_mask_jpg(masks, os.path.join(save_folder, image_name))
                    save_mask_json(save_folder, image_name.split('.')[0], masks, boxes_filt, pred_phrases, image)
                else:
                    print(f'No {text_prompt} found in {image_path}')
                    h, w = image.shape[:2]
                    masks = torch.zeros((1, 1, h, w))
                    save_mask_json(save_folder, image_name.split('.')[0], masks, torch.zeros((0, 4)), [], image)
            print(f'{save_folder} has been processed.')
                    
    print('Done')   


def clean_mask_label():
    target = 'people'
    root = "/home/jiahao/Downloads/data/wildtrack"
    args_str=[  
            "--root", root,
        ]
    
    parser = argparse.ArgumentParser("Grounded-Segment-Anything Demo", add_help=True)
    parser.add_argument("--root", type=str, required=True, help="path to image folder")
    # args = parser.parse_args(args_str)
    args = parser.parse_args()
    ground_mask_dir = os.path.join(args.root, 'masks', 'ground')
    people_mask_dir = os.path.join(args.root, 'masks', 'people')
    image_dir = os.path.join(args.root, 'Image_subsets')
    ground_mask_dirs = [os.path.join(ground_mask_dir, x) for x in sorted(os.listdir(ground_mask_dir))]
    people_mask_dirs = [os.path.join(people_mask_dir, x) for x in sorted(os.listdir(people_mask_dir))]
    for gm_dir, pm_dir in zip(ground_mask_dirs, people_mask_dirs):
        time_step = os.path.basename(gm_dir)
        print('processing:', time_step) 
        for cam_idx in range(1, 8): # camera 1-7
            gm_npy = np.load(os.path.join(gm_dir, f'cam{cam_idx}.npy'))
            pm_npy = np.load(os.path.join(pm_dir, f'cam{cam_idx}.npy'))
            pm_npy[(gm_npy == 1) & (pm_npy > 0)] = 0
            img_p = os.path.join(image_dir, time_step, f'cam{cam_idx}.png')
            img = cv2.imread(img_p)
            # draw the mask for the image
            mask_img = pm_npy
            mask_img = np.stack([mask_img, mask_img, mask_img], axis=-1)
            mask_img = mask_img * 255
            mask_img = mask_img.astype(np.uint8)
            mask_img = cv2.addWeighted(img, 0.5, mask_img, 0.5, 0)
            cv2.imwrite(os.path.join(pm_dir, f'cam{cam_idx}_mask.jpg'), mask_img)#, cv2.cvtColor(mask_img, cv2.COLOR_RGB2BGR))
            np.save(os.path.join(pm_dir, f'cam{cam_idx}.npy'), pm_npy)
            print('saved:', os.path.join(pm_dir, f'cam{cam_idx}_mask.jpg'))




def process_camera(args):
    gm_dir, pm_dir, image_dir, time_step, cam_idx = args
    print(f'Processing: {time_step}, Camera {cam_idx}')
    
    gm_npy = np.load(os.path.join(gm_dir, f'cam{cam_idx}.npy'))
    pm_npy = np.load(os.path.join(pm_dir, f'cam{cam_idx}.npy'))
    pm_npy[(gm_npy == 1) & (pm_npy > 0)] = 0
    
    img_p = os.path.join(image_dir, time_step, f'cam{cam_idx}.png')
    img = cv2.imread(img_p)
    
    mask_img = np.stack([pm_npy, pm_npy, pm_npy], axis=-1) * 255
    mask_img = mask_img.astype(np.uint8)
    mask_img = cv2.addWeighted(img, 0.5, mask_img, 0.5, 0)
    
    cv2.imwrite(os.path.join(pm_dir, f'cam{cam_idx}_mask.jpg'), mask_img)
    np.save(os.path.join(pm_dir, f'cam{cam_idx}.npy'), pm_npy)
    print(f'Saved: {os.path.join(pm_dir, f"cam{cam_idx}_mask.jpg")}')

def clean_mask_label_mt():
    target = 'people'
    root = "/home/jiahao/Downloads/data/wildtrack"
    args_str=[  
            "--root", root,
        ]
    parser = argparse.ArgumentParser("Grounded-Segment-Anything Demo", add_help=True)
    parser.add_argument("--root", type=str, required=True, help="path to image folder")
    # args = parser.parse_args(args_str)
    args = parser.parse_args()
    
    ground_mask_dir = os.path.join(args.root, 'masks', 'ground')
    people_mask_dir = os.path.join(args.root, 'masks', 'people')
    image_dir = os.path.join(args.root, 'Image_subsets')
    
    ground_mask_dirs = [os.path.join(ground_mask_dir, x) for x in sorted(os.listdir(ground_mask_dir))][2000:]
    people_mask_dirs = [os.path.join(people_mask_dir, x) for x in sorted(os.listdir(people_mask_dir))][2000:]
    
    num_processes = cpu_count()
    print(f"Using {num_processes} processes for parallel processing")
    
    with Pool() as pool:
        for gm_dir, pm_dir in zip(ground_mask_dirs, people_mask_dirs):
            time_step = os.path.basename(gm_dir)
            # print(f'\nProcessing: {time_step}')
            
            tasks = [(gm_dir, pm_dir, image_dir, time_step, cam_idx) for cam_idx in range(1, 8)]
            pool.map(process_camera, tasks)
if __name__ == '__main__':
    main()
    # clean_mask_label_mt()