import logging
import pathlib

import torch.utils.data as data
import torchvision.transforms as T
from torchvision.datasets import ImageFolder

from codebase.torchutils.distributed import world_size
from ..utils import get_samplers


_logger = logging.getLogger(__name__)


def get_train_transforms(crop_size, mean, std, is_training):
    pipelines = []
    if is_training:
        pipelines.append(T.RandomResizedCrop(crop_size))
        pipelines.append(T.RandomHorizontalFlip())
    else:
        pipelines.append(T.Resize(int(crop_size/7*8)))
        pipelines.append(T.CenterCrop(crop_size))
    pipelines.append(T.ToTensor())
    pipelines.append(T.Normalize(mean=mean, std=std))
    return T.Compose(pipelines)


def _build_imagenet_loader(root, is_training, image_size, mean, std, batch_size, num_workers, use_webdataset, dataset_len=None):
    transforms = get_train_transforms(image_size, mean, std, is_training)

    dataset = ImageFolder(pathlib.Path(root)/("train" if is_training else "val"), transform=transforms)
    sampler = get_samplers(dataset, is_training)
    loader = data.DataLoader(dataset, batch_size=batch_size,
                             shuffle=(sampler is None),
                             sampler=sampler,
                             num_workers=num_workers,
                             persistent_workers=True,
                             drop_last=is_training)
    _logger.info(f"Loading ImageNet dataset using torchvision from folder"
                 f" with {'trainset' if is_training else 'valset'} (len={len(dataset)})")
    return loader


def build_imagenet_loader(root, image_size, mean, std, batch_size, num_workers,
                          trainset_len, valset_len):
    return _build_imagenet_loader(root, True, image_size, mean, std, batch_size, num_workers, trainset_len),\
        _build_imagenet_loader(root, False, image_size, mean, std, batch_size, num_workers, valset_len)
