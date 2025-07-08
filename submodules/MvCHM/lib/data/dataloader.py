from torch.utils.data import Dataset, DataLoader
from lib.data.wildtrack import Wildtrack
from lib.data.multiviewX import MultiviewX

class MultiviewDataset(Dataset):
    def __init__(self, base:Wildtrack,
                       set_name=str, 
                       split_ratio:float=0.9) -> None:
        super().__init__()
        assert set_name in ['train', 'val', 'all'], 'split mode error'
        self.base = base
        self.__name__ = base.__name__
        self.labels_bbox, self.heatmaps, self.fpaths = self.split(
            set_name, split_ratio, base.labels_bbox, base.heatmaps)
        # camera parameters
        self.intrinsic_matrices, self.extrinsic_matrices = base.intrinsic_matrices, base.extrinsic_matrices

        # depth parameters
        self.depth_map, self.depth_max, self.grid_range = base.depth_map, base.depth_max, base.grid_range
        if isinstance(base, MultiviewX):
            self.depth_min = base.depth_min
     
    def split(self, set_name, split_ratio, labels_bbox, heatmaps):
        """
            Split the labels(annotations), heatmap( the pedestrians' position on the ground ) and 
            the path of image base on set_name. Train set occupies 90%, val set occupies 10%.  
        """
        assert len(labels_bbox) == len(heatmaps) 
        if set_name == 'train':
            if self.base.__name__ == Wildtrack.__name__:
                self.frame_range = range(0, int(self.base.num_frame * split_ratio), 5)
            elif self.base.__name__ == MultiviewX.__name__:
                self.frame_range = range(0, int(self.base.num_frame * split_ratio))
        elif set_name == 'val':
            if self.base.__name__ == Wildtrack.__name__:
                self.frame_range = range(int(self.base.num_frame * split_ratio), int(self.base.num_frame), 5)
            elif self.base.__name__ == MultiviewX.__name__:
                self.frame_range = range(int(self.base.num_frame * split_ratio), int(self.base.num_frame))
        elif set_name == 'all':
            if self.base.__name__ == Wildtrack.__name__:
                self.frame_range = range(0, int(self.base.num_frame), 5)
            elif self.base.__name__ == MultiviewX.__name__:
                self.frame_range = range(0, int(self.base.num_frame))

        if self.base.__name__ == Wildtrack.__name__:
            labels_bbox = [labels_bbox[id] for id, i in enumerate(range(0, int(self.base.num_frame), 5)) if i in self.frame_range]
            heatmaps = [heatmaps[id] for id, i in enumerate(range(0, int(self.base.num_frame), 5)) if i in self.frame_range]
        elif self.base.__name__ == MultiviewX.__name__:
            labels_bbox = [labels_bbox[id] for id, i in enumerate(range(0, int(self.base.num_frame))) if i in self.frame_range]
            heatmaps = [heatmaps[id] for id, i in enumerate(range(0, int(self.base.num_frame))) if i in self.frame_range]
           
        
        fpaths = self.base.get_image_fpaths(self.frame_range)
        self.frame_range = list(self.frame_range)
        return labels_bbox, heatmaps, fpaths
    
    def __len__(self):
        return len(self.frame_range)
    
    def num_classes(self):
        return 1  # only pedestrian
    
    def _load_image(self, index):
        index = self.frame_range[index]
        image_paths = list()
        for vals in self.fpaths.values():
            image_paths.append(vals[index])
        return image_paths

    def _load_annotations(self, index):
        labels_bbox = [ annot for annot in self.labels_bbox[index].values()]
        heatmap = self.heatmaps[index]
        return labels_bbox, heatmap

    def __getitem__(self, index):
        img_path = self._load_image(index)
        labels_bbox, heatmap = self._load_annotations(index)
        return {'image_paths': img_path,
                'annot':labels_bbox,
                'heatmap':heatmap,
                'frame_idx':self.frame_range[index]
                }

    
def collater(data):
    return {'image_paths' : data[0]['image_paths'],
            'annot' : data[0]['annot'],
            'heatmap' : data[0]['heatmap']}

def get_padded_value(h, w, multiple_number=64):
    pad_h = (h + multiple_number - 1) // multiple_number * multiple_number
    pad_w = (w + multiple_number - 1) // multiple_number * multiple_number
    pad_h -= h
    pad_w -= w
    return pad_h, pad_w

