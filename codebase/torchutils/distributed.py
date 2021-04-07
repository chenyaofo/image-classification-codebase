import os
import typing

import torch
import torch.distributed as dist


def patch_OpenPAI_env() -> None:
    """Get distributed informations (e.g., rank, world_size) if run in OpenPAI platform and put thses
    informations into environment variables according the requirements of PyTorch
    """
    DISTRIBUTED_SYNC_PORT = "distributed_sync_port"  # !!! assume the port label for distributed sync is "distributed_sync_port"
    if "PAI_USER_NAME" in os.environ:
        # env "PAI_USER_NAME" exists, that means the job is run by OpenPAI
        # here we copy some essential env variables from OpenPAI preprocessing script
        taskrole = os.environ["PAI_CURRENT_TASK_ROLE_NAME"]
        n_instances = int(os.environ[f"PAI_TASK_ROLE_TASK_COUNT_{taskrole}"])
        if n_instances > 1:  # running with distributed mode in OpenPAI
            # refer to 'https://openpai.readthedocs.io/en/latest/manual/cluster-user/how-to-use-advanced-job-settings.html#environmental-variables-and-port-reservation'
            os.environ['WORLD_SIZE'] = os.environ[f"PAI_TASK_ROLE_TASK_COUNT_{taskrole}"]
            os.environ['MASTER_ADDR'] = os.environ[f'PAI_HOST_IP_{taskrole}_0']
            os.environ['MASTER_PORT'] = os.environ[f'PAI_{taskrole}_0_{DISTRIBUTED_SYNC_PORT}_PORT']
            os.environ["RANK"] = os.environ["PAI_CURRENT_TASK_ROLE_CURRENT_TASK_INDEX"]
            os.environ["LOCAL_RANK"] = str(0)


def distributed_init(backend: str = "nccl", init_method: str = "env://") -> typing.Tuple[str, str, int, int, int, str, str]:
    """Quickly initialize the distributed mode in PyTorch by getting informations from environment variables and
    send these to dist.init_process_group.

    Args:
        backend (str, optional): Refer to torch.distributed.init_process_group. Defaults to "nccl".
        init_method (str, optional): Refer to torch.distributed.init_process_group. Defaults to "env://".

    Returns:
        typing.Tuple[str, str, int, int, int, str, str]: A tuple of (backend, init_method, rank, local_rank, 
        world_size, master_addr, master_port).
    """
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


def is_dist_avail_and_init() -> bool:
    """

    Returns:
        bool: True if distributed mode is initialized correctly, False otherwise.
    """
    return dist.is_available() and dist.is_initialized()


def rank() -> int:
    """

    Returns:
        int: The rank of the current node in distributed system, return 0 if distributed 
        mode is not initialized.
    """
    return dist.get_rank() if is_dist_avail_and_init() else 0


def world_size() -> int:
    """

    Returns:
        int: The world size of the  distributed system, return 1 if distributed mode is not 
        initialized.
    """
    return dist.get_world_size() if is_dist_avail_and_init() else 1


def is_master() -> bool:
    """

    Returns:
        int: True if the rank current node is euqal to 0. Thus it will always return True if 
        distributed mode is not initialized.
    """
    return rank() == 0


def torchsave(obj: typing.Any, f: str) -> None:
    """A simple warp of torch.save. This function is only performed when the current node is the
    master. It will do nothing otherwise. 

    Args:
        obj (typing.Any): The object to save.
        f (str): The output file path.
    """
    if is_master():
        torch.save(obj, f)
