import os
import torch
import numpy as np
from PIL import Image
from PIL import ImageDraw
import matplotlib.pyplot as plt
from errno import EEXIST
from os import makedirs, path

def inverse_sigmoid(x):
    return torch.log(x/(1-x))

def mkdir_p(folder_path):
    # Creates a directory. equivalent to using mkdir -p on the command line
    try:
        makedirs(folder_path)
    except OSError as exc: # Python >2.5
        if exc.errno == EEXIST and path.isdir(folder_path):
            pass
        else:
            raise
        
def get_expon_lr_func(
    lr_init, lr_final, lr_delay_steps=0, lr_delay_mult=1.0, max_steps=1000000
):
    """
    Copied from Plenoxels

    Continuous learning rate decay function. Adapted from JaxNeRF
    The returned rate is lr_init when step=0 and lr_final when step=max_steps, and
    is log-linearly interpolated elsewhere (equivalent to exponential decay).
    If lr_delay_steps>0 then the learning rate will be scaled by some smooth
    function of lr_delay_mult, such that the initial learning rate is
    lr_init*lr_delay_mult at the beginning of optimization but will be eased back
    to the normal learning rate when steps>lr_delay_steps.
    :param conf: config subtree 'lr' or similar
    :param max_steps: int, the number of steps during optimization.
    :return HoF which takes step as input
    """

    def helper(step):
        if step < 0 or (lr_init == 0.0 and lr_final == 0.0):
            # Disable this parameter
            return 0.0
        if lr_delay_steps > 0:
            # A kind of reverse cosine decay.
            delay_rate = lr_delay_mult + (1 - lr_delay_mult) * np.sin(
                0.5 * np.pi * np.clip(step / lr_delay_steps, 0, 1)
            )
        else:
            delay_rate = 1.0
        t = np.clip(step / max_steps, 0, 1)
        log_lerp = np.exp(np.log(lr_init) * (1 - t) + np.log(lr_final) * t)
        return delay_rate * log_lerp

    return helper


def image2gif(path, name='output.gif', duration=500):
    images = []
    for p in sorted(os.listdir(path)):
        full_p = os.path.join(path, p)
        if p.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
            try:
                img = Image.open(full_p)
                # write the image name to the image with fontsize 10
                d = ImageDraw.Draw(img)
                d.text((10, 10), p, fill=(255, 255, 255))
                
                images.append(img)
            except IOError:
                print(f"Error opening {p}. Skipping.")
    
    if images:
        output_path = os.path.join(path, name)
        images[0].save(output_path, save_all=True, append_images=images[1:], duration=duration, loop=0)
        print(f"GIF created: {output_path}")
    else:
        print("No valid images found in the specified directory.")

def rename_file(path):
    for p in sorted(os.listdir(path)):
        if p.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
            fidx = int(p.split('.')[0].split('_')[1])
            new_p = 'epoch_{:04d}.png'.format(fidx)
            full_p = os.path.join(path, p)
            new_full_p = os.path.join(path, new_p)
            os.rename(full_p, new_full_p)
        
if __name__ == '__main__':
    # rename_file('output/superpixel_train')
    image2gif('output/superpixel_train/proteus_depth_constraint', 'depth_constraint.gif', duration=100)