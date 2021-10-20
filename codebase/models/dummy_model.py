import torch
import torch.nn as nn
from .register import MODEL


@MODEL.register
def dummy_model():
    class DymmyModel(nn.Module):
        def __init__(self):
            super(DymmyModel, self).__init__()
            self.linear = nn.Linear(3, 1000)

        def forward(self, x: torch.Tensor):
            x = x.mean(dim=[2, 3], keepdim=False)
            x = self.linear(x)
            return x
    return DymmyModel()
