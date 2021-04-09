import torch.utils.data as data
import torchvision.transforms as T
from torchvision.datasets import CIFAR10, CIFAR100
from torch.utils.data.distributed import DistributedSampler

from .register import DATA
from codebase.torchutils.distributed import is_dist_avail_and_init


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


def get_samplers(trainset, valset):
    if is_dist_avail_and_init():
        train_sampler = DistributedSampler(trainset)
        val_sampler = DistributedSampler(valset, shuffle=False)
    else:
        train_sampler = None
        val_sampler = None
    return train_sampler, val_sampler


@DATA.register
def cifar10(root, mean, std, batch_size, num_workers, **kwargs):
    train_transforms = get_train_transforms(mean, std)
    trainset = CIFAR10(root, train=True, transform=train_transforms, download=True)

    val_transforms = get_val_transforms(mean, std)
    valset = CIFAR10(root, train=False, transform=val_transforms, download=True)

    train_sampler, val_sampler = get_samplers(trainset, valset)
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
def cifar100(root, mean, std, batch_size, num_workers, **kwargs):
    train_transforms = get_train_transforms(mean, std)
    trainset = CIFAR100(root, train=True, transform=train_transforms, download=True)

    val_transforms = get_val_transforms(mean, std)
    valset = CIFAR100(root, train=False, transform=val_transforms, download=True)

    train_sampler, val_sampler = get_samplers(trainset, valset)
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
