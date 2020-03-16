import os

import torch

import typing

import torch.distributed as dist


def init(backend="nccl", init_method="env://"):
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        if dist.is_available():
            rank = int(os.environ["RANK"])
            local_rank = int(os.environ["LOCAL_RANK"])
            world_size = int(os.environ['WORLD_SIZE'])

            master_addr = os.environ["MASTER_ADDR"]
            master_port = os.environ["MASTER_PORT"]

            dist.init_process_group(backend=backend,
                                    init_method=init_method,
                                    world_size=world_size,
                                    rank=rank)
            print(f"Init distributed mode(backend={backend}, "
                  f"init_mothod={master_addr}:{master_port}, "
                  f"rank={rank}, pid={os.getpid()}, world_size={world_size}, "
                  f"is_master={is_master()}).")
            return backend, init_method, rank, local_rank, world_size, master_addr, master_port
        else:
            print("Fail to init distributed because torch.distributed is unavailable.")
        return None, None, 0, 0, 1, None, None


def is_dist_avail_and_init():
    return dist.is_available() and dist.is_initialized()


def rank():
    return dist.get_rank() if is_dist_avail_and_init else 0


def local_rank():
    return int(os.environ["LOCAL_RANK"]) if is_dist_avail_and_init else 0


def world_size():
    return dist.get_world_size() if is_dist_avail_and_init else 1


def is_master():
    return rank() == 0


def torchsave(*args, **kwargs):
    if is_master():
        torch.save(*args, **kwargs)

def dummy_func(*args, **kargs):
    pass

class DummyClass():
    def __getattribute__(self, obj):
        return dummy_func
