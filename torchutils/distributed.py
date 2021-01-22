import os
import typing

import torch
import torch.distributed as dist

def patch_OpenPAI_env():
    DISTRIBUTED_SYNC_PORT = "distributed_sync_port" # !!! assume the port label for distributed sync is "distributed_sync_port"
    if "PAI_USER_NAME" in os.environ:
        # env "PAI_USER_NAME" exists, that means the job is run by OpenPAI
        # here we copy some essential env variables from OpenPAI preprocessing script
        taskrole = os.environ["PAI_CURRENT_TASK_ROLE_NAME"]
        n_instances = int(os.environ[f"PAI_TASK_ROLE_TASK_COUNT_{taskrole}"])
        if n_instances > 1: # running with distributed mode in OpenPAI
            # refer to 'https://openpai.readthedocs.io/en/latest/manual/cluster-user/how-to-use-advanced-job-settings.html#environmental-variables-and-port-reservation'
            os.environ['WORLD_SIZE'] = os.environ[f"PAI_TASK_ROLE_TASK_COUNT_{taskrole}"]
            os.environ['MASTER_ADDR'] = os.environ[f'PAI_HOST_IP_{taskrole}_0']
            os.environ['MASTER_PORT'] = os.environ[f'PAI_{taskrole}_0_{DISTRIBUTED_SYNC_PORT}_PORT']
            os.environ["RANK"] = os.environ["PAI_CURRENT_TASK_ROLE_CURRENT_TASK_INDEX"]
            os.environ["LOCAL_RANK"] = str(0)

def init(backend="nccl", init_method="env://"):
    patch_OpenPAI_env()
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
    return dist.get_rank() if is_dist_avail_and_init() else 0


def local_rank():
    return int(os.environ["LOCAL_RANK"]) if is_dist_avail_and_init() else 0


def world_size():
    return dist.get_world_size() if is_dist_avail_and_init() else 1


def is_master():
    return rank() == 0


def torchsave(obj, f):
    if is_master():
        torch.save(obj, f)
