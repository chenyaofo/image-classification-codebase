import logging
import torch.utils.data as data
import torchvision.transforms as T
from torchvision.datasets import CIFAR10, CIFAR100

from .utils import get_samplers
from .register import DATA


_logger = logging.getLogger(__name__)


def get_train_transforms(mean, std):
    return T.Compose([
        T.RandomCrop(32, padding=4),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize(mean=mean, std=std)
    ])


def get_val_transforms(mean, std):
    return T.Compose([
        T.ToTensor(),
        T.Normalize(mean=mean, std=std)
    ])


def get_vit_train_transforms(mean, std, img_size):
    return T.Compose([
        T.RandomResizedCrop((img_size, img_size), scale=(0.05, 1.0)),
        T.ToTensor(),
        T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])


def get_vit_val_transforms(mean, std, img_size):
    return T.Compose([
        T.Resize((img_size, img_size)),
        T.ToTensor(),
        T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])


def _cifar(root, image_size, mean, std, batch_size, num_workers, is_vit, dataset_builder, **kwargs):
    if is_vit:
        train_transforms = get_vit_train_transforms(mean, std, image_size)
        val_transforms = get_vit_val_transforms(mean, std, image_size)
    else:
        train_transforms = get_train_transforms(mean, std)
        val_transforms = get_val_transforms(mean, std)

    trainset = dataset_builder(root, train=True, transform=train_transforms, download=True)
    valset = dataset_builder(root, train=False, transform=val_transforms, download=True)

    _logger.info(f"Loading {dataset_builder.__name__} dataset with trainset (len={len(trainset)}) and valset (len={len(valset)})")

    train_sampler = get_samplers(trainset, is_training=True)
    val_sampler = get_samplers(valset, is_training=False)

    train_loader = data.DataLoader(trainset, batch_size=batch_size,
                                   shuffle=(train_sampler is None),
                                   sampler=train_sampler,
                                   num_workers=num_workers,
                                   persistent_workers=True)
    val_loader = data.DataLoader(valset, batch_size=batch_size,
                                 shuffle=(val_sampler is None),
                                 sampler=val_sampler,
                                 num_workers=num_workers,
                                 persistent_workers=True)

    return train_loader, val_loader


@DATA.register
def cifar10(root, image_size, mean, std, batch_size, num_workers, is_vit, **kwargs):
    return _cifar(
        root, image_size, mean, std, batch_size, num_workers, is_vit, CIFAR10, **kwargs
    )


@DATA.register
def cifar100(root, image_size, mean, std, batch_size, num_workers, is_vit, **kwargs):
    return _cifar(
        root, image_size, mean, std, batch_size, num_workers, is_vit, CIFAR100, **kwargs
    )
