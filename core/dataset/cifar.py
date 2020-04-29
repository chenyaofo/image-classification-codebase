import torch.utils.data as data
import torchvision.transforms as T
from torchvision.datasets import CIFAR10

from ..transforms import Cutout, CIFAR10Policy
from ..register import DATASET


CIFAR10_MEAN = []
CIFAR10_STD = []


class CIFAR10_D:
    NUM_CLASS = 10
    @staticmethod
    def build_train_loader(args):
        train_transforms = [
            T.RandomCrop(32, padding=4),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize(mean=CIFAR10_MEAN, std=CIFAR10_STD)
        ]
        trainset = CIFAR10(args.data, train=True, transform=train_transforms)
        train_loader = data.DataLoader(trainset, batch_size=args.batch_size,
                                       shuffle=True, num_workers=args.workers)
        return train_loader

    @staticmethod
    def build_val_loader(args):
        val_transforms = [
            T.ToTensor(),
            T.Normalize(mean=CIFAR10_MEAN, std=CIFAR10_STD)
        ]
        valset = CIFAR10(args.data, train=False, transform=val_transforms)
        val_loader = data.DataLoader(valset, batch_size=args.batch_size,
                                     shuffle=False, num_workers=args.workers)
        return val_loader


def get_cifar_train_transform(cutout, autoaugment, mean, std):
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


def get_cifar_val_transform(mean, std):
    return T.Compose([
        T.ToTensor(),
        T.Normalize(mean=mean, std=std),
    ])
