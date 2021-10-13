import math
import warnings
import logging
import pathlib

try:
    import nvidia.dali.types as types
    import nvidia.dali.fn as fn
    from nvidia.dali.plugin.pytorch import DALIGenericIterator, LastBatchPolicy
    from nvidia.dali.pipeline import Pipeline
    import nvidia.dali.tfrecord as tfrec
except ImportError:
    warnings.warn("NVIDIA DALI library is unavailable, cannot load and preprocess dataset with DALI.")

from codebase.torchutils.distributed import world_size, rank
from ..utils import glob_by_suffix


_logger = logging.getLogger(__name__)


def create_dali_pipeline(reader, image_size, batch_size, mean, std, num_workers, local_rank,
                         use_tfrecord, dali_cpu=False, is_training=True):
    # refer to https://github.com/NVIDIA/DALI/blob/54034c4ddd7cfe2b6dda7e8cec5f91ae18f7ad39/docs/examples/use_cases/pytorch/resnet50/main.py
    pipe = Pipeline(batch_size, num_workers, device_id=local_rank)
    with pipe:
        if use_tfrecord:
            images = reader["image"]
            labels = reader["label"]
        else:
            images, labels = reader
        # images, labels = fn.external_source(source=eii, num_outputs=2)
        dali_device = 'cpu' if dali_cpu else 'gpu'
        decoder_device = 'cpu' if dali_cpu else 'mixed'
        # ask nvJPEG to preallocate memory for the biggest sample in ImageNet for CPU and GPU to avoid reallocations in runtime
        device_memory_padding = 211025920 if decoder_device == 'mixed' else 0
        host_memory_padding = 140544512 if decoder_device == 'mixed' else 0
        # ask HW NVJPEG to allocate memory ahead for the biggest image in the data set to avoid reallocations in runtime
        preallocate_width_hint = 5980 if decoder_device == 'mixed' else 0
        preallocate_height_hint = 6430 if decoder_device == 'mixed' else 0

        if is_training:
            images = fn.decoders.image_random_crop(images,
                                                   device=decoder_device, output_type=types.RGB,
                                                   device_memory_padding=device_memory_padding,
                                                   host_memory_padding=host_memory_padding,
                                                   preallocate_width_hint=preallocate_width_hint,
                                                   preallocate_height_hint=preallocate_height_hint,
                                                   random_aspect_ratio=[0.75, 4.0 / 3.0],
                                                   random_area=[0.08, 1.0],
                                                   num_attempts=100)
            images = fn.resize(images,
                               device=dali_device,
                               resize_x=image_size,
                               resize_y=image_size,
                               interp_type=types.INTERP_LINEAR)
            mirror = fn.random.coin_flip(probability=0.5)
        else:
            images = fn.decoders.image(images,
                                       device=decoder_device,
                                       output_type=types.RGB)
            images = fn.resize(images,
                               device=dali_device,
                               size=int(image_size/7*8),
                               mode="not_smaller",
                               interp_type=types.INTERP_LINEAR)
            mirror = False

        images = fn.crop_mirror_normalize(images.gpu(),
                                          dtype=types.FLOAT,
                                          output_layout="CHW",
                                          crop=(image_size, image_size),
                                          mean=[item * 255 for item in mean],
                                          std=[item * 255 for item in std],
                                          mirror=mirror)
        labels = labels.gpu()
        pipe.set_outputs(images, labels)
    return pipe


class DALIWrapper:
    def gen_wrapper(daliiterator):
        for datas in daliiterator:
            inputs = datas[0]["images"]
            targets = datas[0]["targets"].squeeze(-1).long()
            yield inputs, targets
        # daliiterator.reset()

    def __init__(self, daliiterator, _size=None):
        self.daliiterator = daliiterator
        self._size = _size

    def __iter__(self):
        return DALIWrapper.gen_wrapper(self.daliiterator)

    def __len__(self):
        return self._size if self._size is not None else len(self.daliiterator)


def _build_imagenet_dali_loader(root, is_training, image_size, mean, std, batch_size, num_workers,
                                use_tfrecord, dataset_len=None, local_rank=None):

    if use_tfrecord:
        reader = fn.readers.tfrecord(
            path=glob_by_suffix(
                pathlib.Path(root)/("train" if is_training else "val"),
                "*.tfrecord"
            ),
            index_path=glob_by_suffix(
                pathlib.Path(root)/("train" if is_training else "val"),
                "*.idx"
            ),
            features={
                "fname": tfrec.FixedLenFeature((), tfrec.string, ""),
                "image": tfrec.FixedLenFeature((), tfrec.string, ""),
                "label": tfrec.FixedLenFeature([1], tfrec.int64,  -1),
            },
            shard_id=rank(),
            num_shards=world_size(),
            random_shuffle=is_training,
            initial_fill=3000,
            pad_last_batch=False,
            dont_use_mmap=True,  # If set to True, the Loader will use plain file I/O
            # instead of trying to map the file in memory. Mapping provides a small
            # performance benefit when accessing a local file system, but most network
            # file systems, do not provide optimum performance.
            prefetch_queue_depth=2,
            read_ahead=True,
            name="Reader")
    else:
        reader = fn.readers.file(
            file_root=pathlib.Path(root)/("train" if is_training else "val"),
            shard_id=rank(),
            num_shards=world_size(),
            random_shuffle=is_training,
            pad_last_batch=False,
            name="Reader"
        )
    pipe = create_dali_pipeline(reader, image_size, batch_size, mean, std, num_workers, local_rank,
                                use_tfrecord=use_tfrecord, is_training=is_training)
    loader = DALIGenericIterator(pipe,
                                 output_map=["images", "targets"],
                                 #  size = dataset_len if use_webdataset else -1,
                                 auto_reset=True,
                                 last_batch_policy=LastBatchPolicy.DROP if is_training else LastBatchPolicy.PARTIAL,
                                 dynamic_shape=True,
                                 last_batch_padded=True)

    length = None
    if use_tfrecord:
        if is_training:
            length = dataset_len // (world_size() * batch_size)
        else:
            length = math.ceil(dataset_len / (world_size() * batch_size))
        _logger.info(f"Manually set loader.length to {length}")

    loader = DALIWrapper(loader, length)

    _logger.info(f"Loading ImageNet dataset using DALI from {'tfrecord' if use_tfrecord else 'folder'}"
                 f" with {'trainset' if is_training else 'valset'} (len={dataset_len if use_tfrecord else len(reader)})")
    if use_tfrecord:
        _logger.info("Note that the length of tfrecord is reported by user defined config file.")
    return loader


def build_imagenet_dali_loader(root, image_size, mean, std, batch_size, num_workers,
                               use_tfrecord, trainset_len, valset_len, local_rank):
    return _build_imagenet_dali_loader(root, True, image_size, mean, std, batch_size, num_workers,
                                       use_tfrecord, trainset_len, local_rank),\
        _build_imagenet_dali_loader(root, False, image_size, mean, std, batch_size, num_workers,
                                    use_tfrecord, valset_len, local_rank)
