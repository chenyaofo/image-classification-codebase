import pathlib

from torch.utils.data.distributed import DistributedSampler

from codebase.torchutils.distributed import is_dist_avail_and_init


def glob_tars(path):
    tars = list(map(str, pathlib.Path(path).glob('*.tar')))
    tars = sorted(tars)
    return tars


def glob_by_suffix(path, pattern):
    tars = list(map(str, pathlib.Path(path).glob(pattern)))
    tars = sorted(tars)
    return tars


def get_samplers(dataset, is_training):
    return DistributedSampler(dataset, shuffle=is_training) if is_dist_avail_and_init() else None
