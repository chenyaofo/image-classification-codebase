r"""
Modified from https://raw.githubusercontent.com/pytorch/pytorch/v1.7.0/torch/distributed/launch.py

This script aims to quickly start single-node multi-process distributed training with.

"""


import sys
import subprocess
import os
import socket
from argparse import ArgumentParser, REMAINDER


def get_free_port():
    sock = socket.socket()
    sock.bind(('', 0))
    ip, port = sock.getsockname()
    sock.close()
    return port


def parse_args():
    parser = ArgumentParser(description="TorchUitlity distributed training launch "
                                        "helper utility that will spawn up "
                                        "single-node multi-process jobs.")

    parser.add_argument("--gpus", default="0", type=str,
                        help="The GPU ID to use (i.e., CUDA_VISIBLE_DEVICES).")

    # positional
    parser.add_argument("training_script", type=str,
                        help="The full path to the single GPU training "
                             "program/script to be launched in parallel, "
                             "followed by all the arguments for the "
                             "training script")

    # rest from the training program
    parser.add_argument('training_script_args', nargs=REMAINDER)
    return parser.parse_args()


def main():
    args = parse_args()

    n_gpus = len(args.gpus.split(","))

    # here, we specify some command line parameters manually,
    # since in single-node multi-process distributed training, they often are fixed or computable
    args.nnodes = 1
    args.node_rank = 0
    args.nproc_per_node = n_gpus
    args.master_addr = "127.0.0.1"
    args.master_port = get_free_port()
    args.use_env = False
    args.module = False
    args.no_python = False

    # world size in terms of number of processes
    dist_world_size = args.nproc_per_node * args.nnodes

    # set PyTorch distributed related environmental variables
    current_env = os.environ.copy()

    current_env["CUDA_VISIBLE_DEVICES"] = args.gpus

    current_env["MASTER_ADDR"] = args.master_addr
    current_env["MASTER_PORT"] = str(args.master_port)
    current_env["WORLD_SIZE"] = str(dist_world_size)

    processes = []

    if 'OMP_NUM_THREADS' not in os.environ and args.nproc_per_node > 1:
        current_env["OMP_NUM_THREADS"] = str(1)
        # print("*****************************************\n"
        #       "Setting OMP_NUM_THREADS environment variable for each process "
        #       "to be {} in default, to avoid your system being overloaded, "
        #       "please further tune the variable for optimal performance in "
        #       "your application as needed. \n"
        #       "*****************************************".format(current_env["OMP_NUM_THREADS"]))

    for local_rank in range(0, args.nproc_per_node):
        # each process's rank
        dist_rank = args.nproc_per_node * args.node_rank + local_rank
        current_env["RANK"] = str(dist_rank)
        current_env["LOCAL_RANK"] = str(local_rank)

        # spawn the processes
        with_python = not args.no_python
        cmd = []
        if with_python:
            cmd = [sys.executable, "-u"]
            if args.module:
                cmd.append("-m")
        else:
            if not args.use_env:
                raise ValueError("When using the '--no_python' flag, you must also set the '--use_env' flag.")
            if args.module:
                raise ValueError("Don't use both the '--no_python' flag and the '--module' flag at the same time.")

        cmd.append(args.training_script)

        if not args.use_env:
            cmd.append("--local_rank={}".format(local_rank))

        cmd.extend(args.training_script_args)

        process = subprocess.Popen(cmd, env=current_env)
        processes.append(process)

    for process in processes:
        process.wait()
        if process.returncode != 0:
            raise subprocess.CalledProcessError(returncode=process.returncode,
                                                cmd=cmd)


if __name__ == "__main__":
    main()
