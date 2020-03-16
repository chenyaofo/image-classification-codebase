from .MoGA import MoGaA, MoGaB, MoGaC
from .mobilenet import mobilenet_v2
from torchvision.models import resnet18
from .mobilenetv3 import MobileNetV3_Large

def moga_a(args):
    return MoGaA()


def moga_b(args):
    return MoGaB()


def moga_c(args):
    return MoGaC()


def mobilenet_v2_wrap(args):
    return mobilenet_v2()


def resnet18_wrap(args):
    return resnet18()


def build_model(args):
    maps = dict(
        moga_a=moga_a,
        moga_b=moga_b,
        moga_c=moga_c,
        # mobilenet_v2=mobilenet_v2_wrap,
        # resnet18=resnet18_wrap,
    )
    return maps[args.model](args)
