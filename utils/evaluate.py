from typing import List, Tuple, Type, Union
from tqdm.auto import tqdm
import torch
import torch.nn.functional as F
from torch import distributed as dist
from torch import nn
from torch.utils.data import DataLoader


def concat_all_gather(tensors: Union[torch.Tensor, List[torch.Tensor]],
                      num_total_examples: int) -> List:
    '''
    Performs all_gather operation on the provided tensor list.
    '''
    outputs = []
    for tensor in tensors:
        tensor = tensor.cuda()
        tensors_gather = [tensor.clone() for _ in range(dist.get_world_size())]
        dist.all_gather(tensors_gather, tensor)
        output = torch.cat(tensors_gather, dim=0).cpu()
        # truncate the dummy elements added by DistributedInferenceSampler
        outputs.append(output[:num_total_examples])
    return outputs


@torch.inference_mode()
def extract_vid_feature(
        model: Union[nn.Module, Type[nn.Module]], dataloader: DataLoader,
        vid2clip_index: List[int], data_length: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    # In build_dataloader, each original test video is split into a series of equilong clips.
    # During test, we first extact features for all clips
    clip_features, clip_pids, clip_camids, clip_clothes_ids = [], torch.tensor(
        []), torch.tensor([]), torch.tensor([])
    for batch_idx, (vids, batch_pids, batch_camids,
                    batch_clothes_ids) in enumerate(tqdm(dataloader)):
        vids = vids.cuda()
        batch_features = model(vids)
        clip_features.append(batch_features.cpu())
        clip_pids = torch.cat((clip_pids, batch_pids.cpu()), dim=0)
        clip_camids = torch.cat((clip_camids, batch_camids.cpu()), dim=0)
        clip_clothes_ids = torch.cat(
            (clip_clothes_ids, batch_clothes_ids.cpu()), dim=0)
        
    clip_features = torch.cat(clip_features, 0)

    # Use the averaged feature of all clips split from a video as the representation of this original full-length video
    features = torch.zeros(len(vid2clip_index), clip_features.size(1)).cuda()
    clip_features = clip_features.cuda()
    pids = torch.zeros(len(vid2clip_index))
    camids = torch.zeros(len(vid2clip_index))
    clothes_ids = torch.zeros(len(vid2clip_index))
    for i, idx in enumerate(vid2clip_index):
        features[i] = clip_features[idx[0]:idx[1], :].mean(0)
        features[i] = F.normalize(features[i], p=2, dim=0)
        pids[i] = clip_pids[idx[0]]
        camids[i] = clip_camids[idx[0]]
        clothes_ids[i] = clip_clothes_ids[idx[0]]
    features = features.cpu()

    return features, pids, camids, clothes_ids
