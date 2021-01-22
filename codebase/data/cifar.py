import torch.utils.data as data
from torchvision.datasets import cifar
import torchvision.transforms as T
from torchvision.datasets import CIFAR10

from torchutils.distributed import world_size, local_rank
from ..transforms import Cutout, CIFAR10Policy
from ..register import DATASET
from sklearn.utils import shuffle

import nvidia.dali.types as types
import nvidia.dali.ops as ops
from nvidia.dali.pipeline import Pipeline
import torch
import random
from torch.utils.data import Dataset

import numpy

CIFAR10_MEAN = [0.49139968 * 255., 0.48215827 * 255., 0.44653124 * 255.]
CIFAR10_STD = [0.24703233 * 255., 0.24348505 * 255., 0.26158768 * 255.]

class CIFARExternalSource(object):
    def __init__(self, dataset:Dataset, batch_size:int, shuffle:bool=None):
        self.batch_size = batch_size

        self.datas = [torch.tensor(item) for item in dataset.data]
        self.targets = [torch.tensor(item) for item in dataset.targets]

        self.shuffle = dataset.train if shuffle is None else shuffle
    
    def __iter__(self):
        self.i = 0
        self.n = len(self.datas)
        return self

    def __next__(self):
        batch = []
        labels = []
        shuffle_ids = list(range(self.n))
        for _ in range(self.batch_size):
            if self.shuffle and self.i % self.n == 0:
                random.shuffle(shuffle_ids)
            batch.append(self.datas[shuffle_ids[self.i]])
            labels.append(self.targets[shuffle_ids[self.i]])
            self.i = (self.i + 1) % self.n
        return (batch, labels)

class HybridTrainPipe_CIFAR(Pipeline):
    def __init__(self, cifar_dataset, batch_size, num_threads, crop=16, device_id=local_rank(),cpu_only=False, local_rank=0,
                 world_size=1,
                 cutout=0):
        
        super(HybridTrainPipe_CIFAR, self).__init__(batch_size, num_threads, device_id, seed=12 + device_id)

        dali_device = 'cpu' if cpu_only else 'gpu'
        # decoder_device = 'cpu' if cpu_only else 'mixed'

        self.external_source = ops.ExternalSource(
            source=CIFARExternalSource(cifar_dataset, batch_size, shuffle=True),
            num_outputs=2
        )

        # pad 4 pixels per edge
        self.pad = ops.Paste(device=dali_device, ratio=1.25, fill_value=0)
        self.uniform = ops.Uniform(range=(0., 1.))
        self.crop = ops.Crop(device=dali_device, crop_h=crop, crop_w=crop)
        self.cmnp = ops.CropMirrorNormalize(device=dali_device,
                                            output_dtype=types.FLOAT,
                                            output_layout=types.NCHW,
                                            image_type=types.RGB,
                                            mean=CIFAR10_MEAN,
                                            std=CIFAR10_STD
                                            )
        self.coin = ops.CoinFlip(probability=0.5)

    def define_graph(self):
        # self.jpegs = self.input()
        # self.labels = self.input_label()
        # output = self.jpegs
        images, labels = self.external_source()
        # images = self.pad(images)
        # images = self.crop(images, crop_pos_x=self.uniform(), crop_pos_y=self.uniform())
        # images = self.cmnp(images, mirror=self.coin())
        return (images, labels)


# class HybridTestPipe_CIFAR(Pipeline):
#     def __init__(self, batch_size, num_threads, device_id, data_dir, crop, size, local_rank=0, world_size=1):
#         super(HybridTestPipe_CIFAR, self).__init__(batch_size, num_threads, device_id, seed=12 + device_id)
#         self.iterator = iter(CIFAR_INPUT_ITER(batch_size, 'val', root=data_dir))
#         self.input = ops.ExternalSource()
#         self.input_label = ops.ExternalSource()
#         self.cmnp = ops.CropMirrorNormalize(device="gpu",
#                                             output_dtype=types.FLOAT,
#                                             output_layout=types.NCHW,
#                                             image_type=types.RGB,
#                                             mean=[0.49139968 * 255., 0.48215827 * 255., 0.44653124 * 255.],
#                                             std=[0.24703233 * 255., 0.24348505 * 255., 0.26158768 * 255.]
#                                             )

#     def iter_setup(self):
#         (images, labels) = self.iterator.next()
#         self.feed_input(self.jpegs, images, layout="HWC")  # can only in HWC order
#         self.feed_input(self.labels, labels)

#     def define_graph(self):
#         self.jpegs = self.input()
#         self.labels = self.input_label()
#         output = self.jpegs
#         output = self.cmnp(output.gpu())
#         return [output, self.labels]

# class CIFAR10_D:
#     NUM_CLASS = 10
#     @staticmethod
#     def build_train_loader(args):
#         train_transforms = [
#             T.RandomCrop(32, padding=4),
#             T.RandomHorizontalFlip(),
#             T.ToTensor(),
#             T.Normalize(mean=CIFAR10_MEAN, std=CIFAR10_STD)
#         ]
#         trainset = CIFAR10(args.data, train=True, transform=train_transforms)
#         train_loader = data.DataLoader(trainset, batch_size=args.batch_size,
#                                        shuffle=True, num_workers=args.workers)
#         return train_loader

#     @staticmethod
#     def build_val_loader(args):
#         val_transforms = [
#             T.ToTensor(),
#             T.Normalize(mean=CIFAR10_MEAN, std=CIFAR10_STD)
#         ]
#         valset = CIFAR10(args.data, train=False, transform=val_transforms)
#         val_loader = data.DataLoader(valset, batch_size=args.batch_size,
#                                      shuffle=False, num_workers=args.workers)
#         return val_loader


# def get_cifar_train_transform(cutout, autoaugment, mean, std):
#     transforms = [
#         T.RandomCrop(32, padding=4),
#         T.RandomHorizontalFlip(),
#     ]
#     if autoaugment:
#         transforms.append(CIFAR10Policy())
#     if cutout is not None:
#         transforms.append(Cutout(cutout))
#     transforms.append(T.ToTensor())
#     transforms.append(T.Normalize(mean=mean, std=std))
#     return transforms


# def get_cifar_val_transform(mean, std):
#     return T.Compose([
#         T.ToTensor(),
#         T.Normalize(mean=mean, std=std),
#     ])
