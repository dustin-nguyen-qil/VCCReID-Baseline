from torch.utils.data import DataLoader
from datasets.dataset import VideoDataset, TestDataset
from config import CONFIG as config
import os.path as osp
import datasets.spatial_transforms as ST
import datasets.temporal_transforms as TT 
from datasets.samplers import RandomIdentitySampler

def build_transforms():
    spatial_transform_train = ST.Compose([
        ST.Scale((config.DATA.HEIGHT, config.DATA.WIDTH), interpolation=3),
        ST.RandomHorizontalFlip(),
        ST.ToTensor(),
        ST.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ST.RandomErasing(height=config.DATA.HEIGHT,
                         width=config.DATA.WIDTH,
                         probability=config.AUG.RE_PROB)
    ])

    spatial_transform_test = ST.Compose([
        ST.Scale((config.DATA.HEIGHT, config.DATA.WIDTH), interpolation=3),
        ST.ToTensor(),
        ST.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    if config.AUG.TEMPORAL_SAMPLING_MODE == 'tsn':
        temporal_transform_train = TT.TemporalDivisionCrop(
            size=config.AUG.SEQ_LEN)
    elif config.AUG.TEMPORAL_SAMPLING_MODE == 'stride':
        temporal_transform_train = TT.TemporalRandomCrop(
            size=config.AUG.SEQ_LEN, stride=config.AUG.SAMPLING_STRIDE)
    else:
        raise KeyError("Invalid temporal sempling mode '{}'".format(
            config.AUG.TEMPORAL_SAMPLING_MODE))

    temporal_transform_test = None

    return spatial_transform_train, spatial_transform_test, temporal_transform_train, temporal_transform_test

st_train, st_test, tt_train, tt_test = build_transforms()

def build_trainloader():

    """
    Build Train Loader
    """
    if config.TRAIN.TRAIN_MODE == 'standard':
        train_data_path = osp.join(config.DATA.ROOT, config.DATA.DATASET, 'train.pkl')
    else:
        train_data_path = osp.join(config.DATA.ROOT, config.DATA.DATASET, config.TRAIN.TYPE, f'train_{config.TRAIN.TRAIN_MODE}.pkl')

    train = VideoDataset(
        train_data_path, 
        st_train, 
        tt_train
        )
    
    sampler = RandomIdentitySampler(train.dataset, config.DATA.NUM_INSTANCES)

    trainloader = DataLoader(
        train, 
        batch_size=config.DATA.TRAIN_BATCH,
        sampler=sampler,
        num_workers=config.DATA.NUM_WORKERS,
        pin_memory=True, 
        drop_last=True
    )
    return trainloader, train, sampler


def build_testloader():
    """
    Build query and gallery loader
    """
    if config.TEST.TEST_MODE == 'all':
        query_data_path = osp.join(config.DATA.ROOT, config.DATA.TEST_SET, 'query.pkl')
        gallery_data_path = osp.join(config.DATA.ROOT, config.DATA.TEST_SET, 'gallery.pkl')
    else:
        query_data_path = osp.join(config.DATA.ROOT, config.DATA.TEST_SET, config.TEST.TYPE, f'query_{config.TEST.TEST_MODE}.pkl')
        gallery_data_path = osp.join(config.DATA.ROOT, config.DATA.TEST_SET, config.TEST.TYPE, f'gallery_{config.TEST.TEST_MODE}.pkl')

    query = TestDataset(
        query_data_path,
        spatial_transform=st_test,
        temporal_transform=tt_test,
        seq_len=config.AUG.SEQ_LEN,
        stride = config.AUG.SAMPLING_STRIDE
    )

    gallery = TestDataset( 
        gallery_data_path,
        spatial_transform=st_test,
        temporal_transform=tt_test,
        seq_len=config.AUG.SEQ_LEN,
        stride = config.AUG.SAMPLING_STRIDE
    )

    queryloader = DataLoader(
        query, 
        batch_size=config.DATA.TEST_BATCH, 
        num_workers=config.DATA.NUM_WORKERS,
        pin_memory=True, 
        drop_last=False
    )
    galleryloader = DataLoader(
        gallery, 
        batch_size=config.DATA.TEST_BATCH, 
        num_workers=config.DATA.NUM_WORKERS,
        pin_memory=True, 
        drop_last=False
    )

    return queryloader, galleryloader, query, gallery