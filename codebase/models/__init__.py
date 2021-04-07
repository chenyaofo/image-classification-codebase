import torch
import torch.hub

from .register import MODEL


@MODEL.register
def PytorchHub(repo: str, model_name: str, **kwargs):
    return torch.hub.load(repo, model_name, **kwargs)
