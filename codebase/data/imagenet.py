# import os

# import torch.utils.data as data
# import torchvision.transforms as T
# from torchvision.datasets import ImageFolder
# from torch.utils.data.distributed import DistributedSampler


# from codebase.torchutils.distributed import world_size, local_rank

# try:
#     import nvidia.dali.ops as ops
#     import nvidia.dali.types as types
#     from nvidia.dali.pipeline import Pipeline
#     from nvidia.dali.plugin.pytorch import DALIGenericIterator
# except ImportError:
#     raise ImportError("Please install DALI refer to "
#                       "https://docs.nvidia.com/deeplearning/sdk/dali-developer-guide/docs/installation.html.")


# IMAGENET_MEAN = [0.485*255., 0.456*255., 0.406*255.]
# IMAGENET_STD = [0.229*255., 0.224*255., 0.225*255.]


# class ImageNet:
#     NUM_CLASSES = 1000
#     MEAN = [0.485, 0.456, 0.406]
#     MEAN_255 = [0.485*255., 0.456*255., 0.406*255.]
#     STD = [0.229, 0.224, 0.225]
#     STD_255 = [0.229*255., 0.224*255., 0.225*255.]


# class HybridTrainPipe(Pipeline):
#     def __init__(self, batch_size, num_threads, data_dir, crop,
#                  device_id=local_rank(), only_cpu=False):
#         super(HybridTrainPipe, self).__init__(batch_size, num_threads, device_id, seed=12 + device_id)
#         self.input = ops.FileReader(file_root=data_dir, shard_id=local_rank(),
#                                     num_shards=world_size(), random_shuffle=True)
#         # let user decide which pipeline works him bets for RN version he runs
#         dali_device = 'cpu' if only_cpu else 'gpu'
#         decoder_device = 'cpu' if only_cpu else 'mixed'
#         # This padding sets the size of the internal nvJPEG buffers to be able to
#         # handle all images from full-sized ImageNet
#         # without additional reallocations
#         device_memory_padding = 211025920 if decoder_device == 'mixed' else 0
#         host_memory_padding = 140544512 if decoder_device == 'mixed' else 0
#         self.decode = ops.ImageDecoderRandomCrop(device=decoder_device, output_type=types.RGB,
#                                                  device_memory_padding=device_memory_padding,
#                                                  host_memory_padding=host_memory_padding,
#                                                  random_aspect_ratio=[0.8, 1.25],
#                                                  random_area=[0.1, 1.0],
#                                                  num_attempts=100)
#         self.resize = ops.Resize(device=dali_device, resize_x=crop, resize_y=crop,
#                                  interp_type=types.INTERP_TRIANGULAR)
#         self.cmnp = ops.CropMirrorNormalize(device=dali_device,
#                                             output_dtype=types.FLOAT,
#                                             output_layout=types.NCHW,
#                                             crop=(crop, crop),
#                                             image_type=types.RGB,
#                                             mean=IMAGENET_MEAN,
#                                             std=IMAGENET_STD)
#         self.coin = ops.CoinFlip(probability=0.5)

#     def define_graph(self):
#         rng = self.coin()
#         images, labels = self.input(name="Reader")
#         images = self.decode(images)
#         images = self.resize(images)
#         images = self.cmnp(images.gpu(), mirror=rng)
#         return [images, labels]


# class HybridValPipe(Pipeline):
#     def __init__(self, batch_size, num_threads, data_dir, crop, size, device_id=local_rank(), only_cpu=False):
#         super(HybridValPipe, self).__init__(batch_size, num_threads, device_id, seed=12 + device_id)
#         self.input = ops.FileReader(file_root=data_dir, shard_id=local_rank(),
#                                     num_shards=world_size(), random_shuffle=False)
#         dali_device = 'cpu' if only_cpu else 'gpu'
#         decoder_device = 'cpu' if only_cpu else 'mixed'
#         self.decode = ops.ImageDecoder(device=decoder_device, output_type=types.RGB)
#         self.resize = ops.Resize(device=dali_device, resize_shorter=size, interp_type=types.INTERP_TRIANGULAR)
#         self.cmnp = ops.CropMirrorNormalize(device=dali_device,
#                                             output_dtype=types.FLOAT,
#                                             output_layout=types.NCHW,
#                                             crop=(crop, crop),
#                                             image_type=types.RGB,
#                                             mean=IMAGENET_MEAN,
#                                             std=IMAGENET_STD)

#     def define_graph(self):
#         images, labels = self.input(name="Reader")
#         images = self.decode(images)
#         images = self.resize(images)
#         images = self.cmnp(images)
#         return [images, labels]


# class ImageNet:
#     NUM_CLASS = 1000

#     @staticmethod
#     def build_train_loader(args):
#         train_pipe = HybridTrainPipe(batch_size=args.batch_size,
#                                      num_threads=args.num_workers,
#                                      data_dir=os.path.join(args.data, "train"),
#                                      crop=args.image_size)
#         train_pipe.build()
#         train_loader = DALIGenericIterator(pipelines=train_pipe,
#                                            output_map=["sample", "label"],
#                                            reader_name="Reader",
#                                            auto_reset=True)
#         return train_loader

#     @staticmethod
#     def build_val_loader(args):
#         val_pipe = HybridValPipe(batch_size=args.batch_size,
#                                  num_threads=args.num_workers,
#                                  data_dir=os.path.join(args.data, "val"),
#                                  crop=args.image_size,
#                                  size=int(args.image_size/0.875))
#         val_pipe.build()
#         val_loader = DALIGenericIterator([val_pipe],
#                                          val_pipe.epoch_size("Reader") / world_size())
#         return val_loader


def build_imagenet_loader(args):
    return None, None
    # return ImageNet.build_train_loader(args), ImageNet.build_val_loader(args)
