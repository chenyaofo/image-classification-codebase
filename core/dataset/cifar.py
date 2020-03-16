import torch.utils.data as data
import torchvision.transforms as T
from torchvision.datasets import CIFAR10, CIFAR100

from ..transforms import Cutout, CIFAR10Policy
from ..register import DATASET


def get_cifar_train_transform(cutout,autoaugment,mean,std):
    transforms = [
        T.RandomCrop(32, padding=4),
        T.RandomHorizontalFlip(),
    ]
    if autoaugment:
        transforms.append(CIFAR10Policy())
    if cutout is not None:
        transforms.append(Cutout(cutout))
    transforms.append(T.ToTensor())
    transforms.append(T.Normalize(mean=mean, std=std))
    return transforms


def get_cifar_val_transform(mean,std):
    return T.Compose([
        T.ToTensor(),
        T.Normalize(mean=mean, std=std),
    ])

def get_cifar_trainset(dataset=CIFAR10, root=".data", transform=None, target_transform=None, download=True):
    ds = dataset(root=root, train=True, transform=transform,
                 target_transform=target_transform, download=download)
    return ds


def get_cifar_valset(dataset=CIFAR10, root=".data", transform=None, target_transform=None, download=True):
    ds = dataset(root=root, train=False, transform=transform,
                 target_transform=target_transform, download=download)
    return ds


def get_cifar_train_loader(dataset=CIFAR10, root=".data",
                           transform=None, target_transform=None, download=True,
                           batch_size=128, num_workers=4):
    ds = get_cifar_trainset(dataset, root, transform,
                            target_transform, download)
    return data.DataLoader(dataset=ds, shuffle=True, batch_size=batch_size, num_workers=num_workers)


def get_cifar_val_loader(dataset=CIFAR10, root=".data",
                         transform=None, target_transform=None, download=True,
                         batch_size=128, num_workers=4):
    ds = get_cifar_valset(dataset, root, transform,
                          target_transform, download)
    return data.DataLoader(dataset=ds, shuffle=False, batch_size=batch_size, num_workers=num_workers)

def _cifar(dataset_type:CIFAR10, cfg):
    ROOT =cfg["dataset.root"]
    DATASET_MEAN =  cfg["dataset.mean"]
    DATASET_STD =  cfg["dataset.std"]
    CUTOUT = cfg["transforms.cutout"]
    AUTOAUGMENT = cfg["transforms.autoaugment"]
    BATCH_SIZE = cfg["strategy.batch_size"]
    NUM_WORKERS = cfg["machine.num_workers"]

    train_transforms = get_cifar_train_transform(CUTOUT,AUTOAUGMENT,DATASET_MEAN, DATASET_STD)
    val_transforms = get_cifar_val_transform(DATASET_MEAN, DATASET_STD)

    trainset = dataset_type(root=ROOT, train=True, transform=train_transforms, download=True)
    valset = dataset_type(root=ROOT, train=False, transform=val_transforms, download=True)

    train_loader = data.DataLoader(dataset=trainset, shuffle=True, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS)
    val_loader = data.DataLoader(dataset=valset, shuffle=False, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS)

    return train_loader,val_loader

@DATASET.register
def cifar10(cfg):
    return _cifar(CIFAR10, cfg)

@DATASET.register
def cifar100(cfg):
    return _cifar(CIFAR100, cfg)