from .native import build_imagenet_loader
from .dali import build_imagenet_dali_loader

from ..register import DATA


@DATA.register
def imagenet2012(root, image_size, mean, std, batch_size, num_workers, use_dali,
                 use_webdataset, use_tfrecord, trainset_len, valset_len, local_rank, **kwargs):
    if use_dali:
        return build_imagenet_dali_loader(root, image_size, mean, std, batch_size, num_workers,
                                          use_webdataset, use_tfrecord, trainset_len, valset_len, local_rank)
    else:
        return build_imagenet_loader(root, image_size, mean, std, batch_size, num_workers,
                                     use_webdataset, trainset_len, valset_len)
