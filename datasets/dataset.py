import pickle
import torch
from torch.utils.data import Dataset

from datasets.utils import *

class VideoDataset(Dataset):
    """Video Person ReID Dataset.
    Note:
        Batch data has shape N x C x T x H x W
    Args:
        dataset (list): List with items (img_paths, pid, camid)
        temporal_transform (callable, optional): A function/transform that  takes in a list of frame indices
            and returns a transformed version
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        loader (callable, optional): A function to load an video given its path and frame indices.
    """

    def __init__(self,
                 data_path,
                 spatial_transform=None,
                 temporal_transform=None,
                 get_loader=get_default_video_loader,
                ):
        data, self.num_pids = self.read_dataset(data_path)
        self.spatial_transform = spatial_transform
        self.temporal_transform = temporal_transform
        self.loader = get_loader
        self.dataset = densesampling_for_trainingset(data)
            

    def __len__(self):
        return len(self.dataset)
    
    def read_dataset(self, data_path):
        with open(data_path, 'rb') as f:
            content = pickle.load(f)
        data = content['data']
        num_pids = content['num_pids']
        return data, num_pids

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
            img_paths = tracklet['img_paths']
            pid = tracklet['p_id']
            camid = tracklet['cam_id']
            clothes_id = tracklet['clothes_id']
            xcs = tracklet['shape_1024']
            betas = tracklet['betas']

        Returns:
            tuple: (clip, pid, camid) where pid is identity of the clip.
            
        """
        tracklet = self.dataset[index]
        (pid, camid, clothes_id, img_paths) = tracklet

        if self.temporal_transform is not None:
            img_paths_tt = self.temporal_transform(img_paths)
        
        clip = self.loader(img_paths_tt)

        if self.spatial_transform is not None:
            self.spatial_transform.randomize_parameters()
            clip = [self.spatial_transform(img) for img in clip]

        # trans T x C x H x W to C x T x H x W
        clip = torch.stack(clip, 0).permute(1, 0, 2, 3)
        

        return clip, pid, camid, clothes_id
    
        
        
class TestDataset(Dataset):
    """Video Person ReID Dataset.
    Note:
        Batch data has shape N x C x T x H x W
    Args:
        dataset (list): List with items (img_paths, pid, camid)
        temporal_transform (callable, optional): A function/transform that  takes in a list of frame indices
            and returns a transformed version
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        loader (callable, optional): A function to load an video given its path and frame indices.
    """

    def __init__(self,
                 data_path,
                 spatial_transform=None,
                 temporal_transform=None,
                 get_loader=get_default_video_loader,
                 seq_len: int=16,
                 stride: int=4
                ):
        data, self.num_pids = self.read_dataset(data_path)
        self.spatial_transform = spatial_transform
        self.temporal_transform = temporal_transform
        self.loader = get_loader
        self.dataset, self.vid2clip_index = recombination_for_testset(data, seq_len, stride)

    def __len__(self):
        return len(self.dataset)
    
    def read_dataset(self, data_path):
        with open(data_path, 'rb') as f:
            content = pickle.load(f)
        data = content['data']
        num_pids = content['num_pids']
        return data, num_pids

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
            img_paths = tracklet['img_paths']
            pid = tracklet['p_id']
            camid = tracklet['cam_id']
            clothes_id = tracklet['clothes_id']
            xcs = tracklet['shape_1024']
            betas = tracklet['betas']

        Returns:
            tuple: (clip, pid, camid) where pid is identity of the clip.
            
        """
        img_paths, pid, camid, clothes_id = self.dataset[index]     
        
        if self.temporal_transform is not None:
            img_paths = self.temporal_transform(img_paths)
        
        clip = self.loader(img_paths)

        if self.spatial_transform is not None:
            self.spatial_transform.randomize_parameters()
            clip = [self.spatial_transform(img) for img in clip]

        # trans T x C x H x W to C x T x H x W
        clip = torch.stack(clip, 0).permute(1, 0, 2, 3)
        return clip, pid, camid, clothes_id
        