import torch
import torch.hub

from .register import MODEL


@MODEL.register
def PyTorchHub(repo: str, name: str, **kwargs):
    return torch.hub.load(repo, name, **kwargs)
