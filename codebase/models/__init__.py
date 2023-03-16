import torch
import torch.hub

from torchvision.models import resnet18, resnet50
from torchvision.models import mobilenet_v2, shufflenet_v2_x1_0
from torchvision.models import vit_b_16, swin_t
from .dummy_model import dummy_model

from .register import MODEL


@MODEL.register
def PyTorchHub(repo: str, name: str, **kwargs):
    return torch.hub.load(repo, name, **kwargs)


MODEL.register(resnet18)
MODEL.register(resnet50)
MODEL.register(mobilenet_v2)
MODEL.register(shufflenet_v2_x1_0)
MODEL.register(vit_b_16)
MODEL.register(swin_t)
