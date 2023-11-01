from torch.utils.data import DataLoader
from lib.core.data_loader import CheckpointDataLoader
from lib.datasets import BaseDataset, MixedDataset
from tqdm import tqdm


def get_dataloaders(cfg=None):

    if cfg is None:
        train_bs = 32
        num_workers = 4
    else:
        train_bs = cfg.TRAIN.BATCH_SIZE
        num_workers = cfg.NUM_WORKERS
        print('Num of data loading workers:', num_workers)

    crop_size = cfg.IMG_RES
    dataset_list = cfg.DATASET.LIST
    partition = cfg.DATASET.PARTITION

    train = MixedDataset(dataset_list, partition, 
        is_train=True, use_augmentation=True, normalization=True, crop_size=crop_size, max_cache_size=cfg.MAX_CACHE_DIM)
    train_loader = CheckpointDataLoader(train, batch_size=train_bs, num_workers=num_workers)

    valid_set = 'bedlam_val'
    test = BaseDataset(valid_set, is_train=False, use_augmentation=False, 
        normalization=True, cropped=False, crop_size=crop_size, max_cache_size=cfg.MAX_CACHE_DIM)
    test_loader = DataLoader(test, batch_size=32, shuffle=False, num_workers=4)


    return [train_loader, test_loader]


