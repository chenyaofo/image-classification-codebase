
import sys
import pathlib
from dataclasses import dataclass
from typing import List, Optional

from pyhocon import ConfigFactory, ConfigTree

from codebase.torchutils.common import get_free_port
from codebase.torchutils.typed_args import TypedArgs, add_argument


@dataclass
class Args(TypedArgs):
    output_dir: str = add_argument("-o", "--output-dir", default="")
    conf: str = add_argument("--conf", default="")
    modifications: List[str] = add_argument("-M", nargs='+', help="list")

    world_size: int = add_argument("--world-size", default=1)
    dist_backend: str = add_argument("--dist-backend", default="nccl")
    dist_url: Optional[str] = add_argument("--dist-url", default=None)
    node_rank: int = add_argument("--node-rank", default=0)


def get_args():
    args, _ = Args.from_known_args(sys.argv)
    args.output_dir = pathlib.Path(args.output_dir)

    if args.dist_url is None:
        args.dist_url = f"tcp://127.0.0.1:{get_free_port()}"

    args.conf = ConfigFactory.parse_file(args.conf)
    args.output_dir.mkdir(parents=True, exist_ok=args.conf.get_bool("auto_resume"))

    if args.modifications is not None:
        for modifition in args.modifications:
            key, value = modifition.split("=")
            value = eval(value)
            if key not in args.conf:
                raise ValueError(f"Key '{key}'' is not in the config tree!")
            args.conf.put(key, value)
    return args
