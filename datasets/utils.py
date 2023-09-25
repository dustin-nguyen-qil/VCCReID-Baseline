
import os
from PIL import Image
import numpy as np
import math



def read_image(img_path):
    """Keep reading image until succeed.
    This can avoid IOError incurred by heavy IO process."""
    got_img = False
    if not os.path.exists(img_path):
        raise IOError("{} does not exist".format(img_path))
    while not got_img:
        try:
            img = Image.open(img_path).convert('RGB')
            got_img = True
        except IOError:
            print(
                "IOError incurred when reading '{}'. Will redo. Don't worry. Just chill."
                .format(img_path))
            pass
    return img


def pil_loader(path):
    from PIL import ImageFile
    ImageFile.LOAD_TRUNCATED_IMAGES = True  
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')


def get_default_image_loader():
    return pil_loader


def video_loader(img_paths, image_loader):
    video = []
    for image_path in img_paths:
        if os.path.exists(image_path):
            video.append(image_loader(image_path))
        else:
            return video

    return video


def get_default_video_loader(img_paths):
    image_loader = get_default_image_loader()
    return video_loader(img_paths, image_loader=image_loader)
    # return functools.partial(video_loader(img_paths), image_loader=image_loader)

def densesampling_for_trainingset(dataset, sampling_step=64):
    ''' Split all videos in training set into lots of clips for dense sampling.

    Args:
        dataset (list): input dataset, each video is organized as (img_paths, pid, camid, clothes_id)
        sampling_step (int): sampling step for dense sampling

    Returns:
        new_dataset (list): output dataset
    '''
    new_dataset = []
    for item in dataset:
        pid = item['p_id']
        camid = item['cam_id']
        clothes_id = item['clothes_id']
        img_paths = item['img_paths']
        
        if sampling_step != 0:
            num_sampling = len(img_paths) // sampling_step
            if num_sampling == 0:
                new_dataset.append((pid, camid, clothes_id, img_paths))
            else:
                for idx in range(num_sampling):
                    if idx == num_sampling - 1:
                        new_dataset.append(
                            (pid, camid, clothes_id, img_paths[idx * sampling_step:]))
                    else:
                        new_dataset.append(
                            (pid, camid, clothes_id,
                             img_paths[idx * sampling_step:(idx + 1) * sampling_step], 
                            ))
        else:
            new_dataset.append((pid, camid, clothes_id, img_paths))

    return new_dataset

def recombination_for_testset(dataset, seq_len=16, stride=4):
        ''' Split all videos in test set into lots of equilong clips.

        Args:
            dataset (list): input dataset, each video is organized as (img_paths, pid, camid, clothes_id)
            seq_len (int): sequence length of each output clip
            stride (int): temporal sampling stride

        Returns:
            new_dataset (list): output dataset with lots of equilong clips
            vid2clip_index (list): a list contains the start and end clip index of each original video
        '''
        new_dataset = []
        vid2clip_index = np.zeros((len(dataset), 2), dtype=int)
        for idx, item in enumerate(dataset):
            pid = item['p_id']
            img_paths = item['img_paths']
            camid = item['cam_id']
            clothes_id = item['clothes_id']
            # start index
            vid2clip_index[idx, 0] = len(new_dataset)
            # process the sequence that can be divisible by seq_len*stride
            for i in range(len(img_paths) // (seq_len * stride)):
                for j in range(stride):
                    begin_idx = i * (seq_len * stride) + j
                    end_idx = (i + 1) * (seq_len * stride)
                    clip_paths = img_paths[begin_idx:end_idx:stride]
                    assert (len(clip_paths) == seq_len)
                    new_dataset.append((clip_paths, pid, camid, clothes_id))
            # process the remaining sequence that can't be divisible by seq_len*stride
            if len(img_paths) % (seq_len * stride) != 0:
                # reducing stride
                new_stride = (len(img_paths) % (seq_len * stride)) // seq_len
                for i in range(new_stride):
                    begin_idx = len(img_paths) // (seq_len * stride) * (
                        seq_len * stride) + i
                    end_idx = len(img_paths) // (seq_len * stride) * (
                        seq_len * stride) + seq_len * new_stride
                    clip_paths = img_paths[begin_idx:end_idx:new_stride]
                    assert (len(clip_paths) == seq_len)
                    new_dataset.append((clip_paths, pid, camid, clothes_id))
                # process the remaining sequence that can't be divisible by seq_len
                if len(img_paths) % seq_len != 0:
                    clip_paths = img_paths[len(img_paths) // seq_len *
                                           seq_len:]
                    # loop padding
                    while len(clip_paths) < seq_len:
                        for index in clip_paths:
                            if len(clip_paths) >= seq_len:
                                break
                            clip_paths.append(index)
                    assert (len(clip_paths) == seq_len)
                    new_dataset.append((clip_paths, pid, camid, clothes_id))
            # end index
            vid2clip_index[idx, 1] = len(new_dataset)
            assert ((vid2clip_index[idx, 1] -
                     vid2clip_index[idx, 0]) == math.ceil(
                         len(img_paths) / seq_len))

        return new_dataset, vid2clip_index.tolist()