import torch.utils.data as data
import torchvision.transforms as T
from torchvision.datasets import CIFAR10
from torch.utils.data.distributed import DistributedSampler

from .register import DATA
from codebase.torchutils.distributed import is_dist_avail_and_init


@DATA.register
def cifar10(root, mean, std, batch_size, num_workers, **kwargs):
    train_transforms = T.Compose([
        T.RandomCrop(32, padding=4),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize(mean=mean, std=std)
    ])
    trainset = CIFAR10(root, train=True, transform=train_transforms, download=True)

    val_transforms = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=mean, std=std)
    ])
    valset = CIFAR10(root, train=False, transform=val_transforms, download=True)

    if is_dist_avail_and_init():
        train_sampler = DistributedSampler(trainset)
        val_sampler = DistributedSampler(valset, shuffle=False)
    else:
        train_sampler = None
        val_sampler = None

    train_loader = data.DataLoader(trainset, batch_size=batch_size,
                                   shuffle=(train_sampler is None),
                                   sampler=train_sampler,
                                   num_workers=num_workers)
    val_loader = data.DataLoader(valset, batch_size=batch_size,
                                 shuffle=(val_sampler is None),
                                 sampler=val_sampler,
                                 num_workers=num_workers)

    return train_loader, val_loader
