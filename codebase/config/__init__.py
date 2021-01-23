import sys
from dataclasses import dataclass

from torchutils.typed_args import TypedArgs, add_argument


@dataclass
class Args(TypedArgs):
    # dataset
    data: str = add_argument('--data', default="", help='path to dataset')
    num_classes: int = add_argument("--num_classes", default=1000)
    model: str = add_argument('--model', metavar='ARCH', default='moga_a')

    dali: bool = add_argument('--dali', action='store_true', default=False)
    max_epochs: int = add_argument('--max_epochs', default=150)
    criterion: str = add_argument('--criterion', default='cross_entropy')

    dataset: str = add_argument('--dataset', default='imagenet')

    optimizer: str = add_argument('--optimizer', default='sgd')
    lr: float = add_argument('--lr', default=0.1)
    momentum: float = add_argument('--momentum', default=0.9)
    weight_decay: float = add_argument('--weight_decay', default=1e-5)

    scheduler: str = add_argument('--scheduler', default='warmup_cosine')
    warmup_epochs: int = add_argument('--warmup_epochs', default=5)
    eta_min: float = add_argument('--eta_min', default=1e-3)

    batch_size: int = add_argument("-b", "--batch_size", default=64)
    num_workers: int = add_argument('-j', '--workers', default=8, metavar='N',
                                    help='number of data loading workers (default: 4)')

    image_size: int = add_argument("--image_size", default=224)

    # resume: str = add_argument("--resume", default=None)

    report_freq: int = add_argument("--report_freq", default=10)

    exponential_lr_lambda: float = add_argument('--exponential_lr_lambda', default=0.97)
    dropout: float = add_argument('--dropout', default=0.2)
    bn_momentum: float = add_argument('--bn_momentum', default=0.1)

    amp: bool = add_argument("--amp", action="store_true",
                             help="Use native PyTorch auto mixed precision (AMP).")
    auto_resume: bool = add_argument("--auto_resume", action="store_true",
                                     help="Automatically load the checkpoint in the output directory.")


args, _ = Args.from_known_args(sys.argv)
