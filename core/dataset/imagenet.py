import os

import torch.utils.data as data
import torchvision.transforms as T
from torchvision.datasets import ImageFolder
from torch.utils.data.distributed import DistributedSampler
from nvidia.dali.plugin.pytorch import DALIClassificationIterator

from utils.distributed import world_size, local_rank

from .dali import HybridTrainPipe, HybridValPipe

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def build_imagenet_loader(args):
    if args.dali:
        train_pipe = HybridTrainPipe(batch_size=args.batch_size, num_threads=args.num_workers,
                                    data_dir=os.path.join(args.data, "train"), crop=args.image_size)
        train_pipe.build()
        train_loader = DALIClassificationIterator([train_pipe], train_pipe.epoch_size("Reader") / world_size())

        val_pipe = HybridValPipe(batch_size=args.batch_size, num_threads=args.num_workers,
                                data_dir=os.path.join(args.data, "val"), crop=args.image_size,
                                size=int(args.image_size/0.875))
        val_pipe.build()
        val_loader = DALIClassificationIterator([val_pipe], val_pipe.epoch_size("Reader") / world_size())
    else:
        trainset = ImageFolder(
            root=os.path.join(args.data, "train"),
            transform=T.Compose([
                T.RandomResizedCrop(args.image_size),
                T.RandomHorizontalFlip(),
                T.ToTensor(),
                T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            ])
        )
        train_sampler = DistributedSampler(trainset)

        train_loader = data.DataLoader(
            dataset=trainset,
            batch_size=args.batch_size,
            sampler=train_sampler,
            num_workers=args.num_workers,
        )

        valset = ImageFolder(
            root=os.path.join(args.data, "val"),
            transform=T.Compose([
                T.Resize(int(args.image_size/0.875)),
                T.CenterCrop(args.image_size),
                T.ToTensor(),
                T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            ])
        )
        val_sampler = DistributedSampler(valset, shuffle=False)
        val_loader = data.DataLoader(
            dataset=valset,
            batch_size=args.batch_size,
            sampler=val_sampler,
            num_workers=args.num_workers,
        )
    return train_loader, val_loader
