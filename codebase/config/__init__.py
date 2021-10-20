
import re
import sys
import pathlib
from dataclasses import dataclass
from typing import List, Optional

from pyhocon import ConfigFactory, ConfigTree

from codebase.torchutils.common import get_free_port
from codebase.torchutils.typed_args import TypedArgs, add_argument


def is_valid_domain(value):
    pattern = re.compile(
        r'^(([a-zA-Z]{1})|([a-zA-Z]{1}[a-zA-Z]{1})|'
        r'([a-zA-Z]{1}[0-9]{1})|([0-9]{1}[a-zA-Z]{1})|'
        r'([a-zA-Z0-9][-_.a-zA-Z0-9]{0,61}[a-zA-Z0-9]))\.'
        r'([a-zA-Z]{2,13}|[a-zA-Z0-9-]{2,30}.[a-zA-Z]{2,3})$'
    )
    return True if pattern.match(value) else False


def is_valid_ip(str):
    p = re.compile('^((25[0-5]|2[0-4]\d|[01]?\d\d?)\.){3}(25[0-5]|2[0-4]\d|[01]?\d\d?)$')
    return True if p.match(str) else False


@dataclass
class Args(TypedArgs):
    output_dir: str = add_argument("-o", "--output-dir", default="")
    conf: str = add_argument("--conf", default="")
    modifications: List[str] = add_argument("-M", nargs='+', help="list")

    world_size: int = add_argument("--world-size", default=1)
    dist_backend: str = add_argument("--dist-backend", default="nccl")
    dist_url: Optional[str] = add_argument("--dist-url", default=None)
    node_rank: int = add_argument("--node-rank", default=0)


def get_args(argv=sys.argv):
    args, _ = Args.from_known_args(argv)
    args.output_dir = pathlib.Path(args.output_dir)

    if args.dist_url is None:
        args.dist_url = f"tcp://127.0.0.1:{get_free_port()}"
    elif is_valid_domain(args.dist_url) or is_valid_ip(args.dist_url):
        args.dist_url = f"tcp://{args.dist_url}:{get_free_port()}"

    args.conf = ConfigFactory.parse_file(args.conf)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.modifications is not None:
        for modifition in args.modifications:
            key, value = modifition.split("=")
            try:
                eval_value = eval(value)
            except Exception:
                eval_value = value
            if key not in args.conf:
                raise ValueError(f"Key '{key}'' is not in the config tree!")
            args.conf.put(key, eval_value)
    return args
