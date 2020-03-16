import sys
from dataclasses import dataclass

from utils.typed_args import TypedArgs, add_argument


@dataclass
class Args(TypedArgs):
    data: str = add_argument('--data', metavar='DIR', help='path to dataset')
    model: str = add_argument('--model', metavar='ARCH', default='moga_a')

    dali:bool = add_argument('--dali',action='store_true', default=False)
    max_epochs: int = add_argument('--max_epochs', default=150)
    criterion: str = add_argument('--criterion', default='ce')


    dataset: str = add_argument('--dataset', default='imagenet')

    optimizer: str = add_argument('--optimizer', default='sgd')
    lr: float = add_argument('--lr', default=0.1)
    momentum: float = add_argument('--momentum', default=0.9)
    weight_decay: float = add_argument('--weight_decay', default=3e-5)

    scheduler: str = add_argument('--scheduler', default='warmup_cosine')
    warmup_epochs: int = add_argument('--warmup_epochs', default=5)
    eta_min: float = add_argument('--eta_min', default=1e-3)


    batch_size: int = add_argument("-b", "--batch_size", default=64)
    num_workers: int = add_argument('-j', '--workers', default=8, metavar='N',
                                    help='number of data loading workers (default: 4)')

    image_size: int = add_argument("--image_size", default=224)

    resume: str = add_argument("--resume", default=None)

    report_freq: int = add_argument("--report_freq", default=10)


args = Args.from_known_args(sys.argv)
