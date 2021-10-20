import torch
import torch.hub

from torchvision.models import resnet18, resnet50
from .dummy_model import dummy_model

from .register import MODEL


@MODEL.register
def PyTorchHub(repo: str, name: str, **kwargs):
    return torch.hub.load(repo, name, **kwargs)


MODEL.register(resnet18)
MODEL.register(resnet50)
