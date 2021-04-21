from collections import defaultdict
from typing import Any, Counter, DefaultDict, Dict, Optional, Tuple, Union, List
import copy
import torch
import torch.nn as nn
from torch import Tensor

from .analysis import JitModelAnalysis
from .handle import (
    Handle,
    addmm_madds_jit,
    bmm_madds_jit,
    conv_madds_jit,
    einsum_madds_jit,
    elementwise_madds_counter,
    linear_madds_jit,
    matmul_madds_jit,
    norm_madds_counter,
)


# A dictionary that maps supported operations to their madds count jit handles.

DENSE_OPS: Dict[str, Handle] = {
    "aten::addmm": addmm_madds_jit,
    "aten::bmm": bmm_madds_jit,
    "aten::_convolution": conv_madds_jit,
    "aten::einsum": einsum_madds_jit,
    "aten::matmul": matmul_madds_jit,
    "aten::linear": linear_madds_jit,
}

NORM_OPS: Dict[str, Handle] = {
    "aten::batch_norm": norm_madds_counter(1),
    "aten::group_norm": norm_madds_counter(2),
    "aten::layer_norm": norm_madds_counter(2),
    "aten::instance_norm": norm_madds_counter(1),
}

ELEMENTWISE_OPS: Dict[str, Handle] = {
    "aten::upsample_nearest2d": elementwise_madds_counter(0, 1),
    "aten::upsample_bilinear2d": elementwise_madds_counter(0, 4),
    "aten::adaptive_avg_pool2d": elementwise_madds_counter(1, 0),
    "aten::grid_sampler": elementwise_madds_counter(0, 4),  # assume bilinear
}


class MaddsAnalysis(JitModelAnalysis):
    """
    Provides access to per-submodule model madds count obtained by
    tracing a model with pytorch's jit tracing functionality. By default,
    comes with standard madds counters for a few common operators.
    Note that:
        1. madds is not a well-defined concept. We just produce our best estimate.
        2. We count one fused multiply-add as one madds.
    Handles for additional operators may be added, or the default ones
    overwritten, using the ``.set_op_handle(name, func)`` method.
    See the method documentation for details.
    madds counts can be obtained as:
    * ``.total(module_name="")``: total madds count for the module
    * ``.by_operator(module_name="")``: madds counts for the module, as a Counter
      over different operator types
    * ``.by_module()``: Counter of madds counts for all submodules
    * ``.by_module_and_operator()``: dictionary indexed by descendant of Counters
      over different operator types
    An operator is treated as within a module if it is executed inside the
    module's ``__call__`` method. Note that this does not include calls to
    other methods of the module or explicit calls to ``module.forward(...)``.
    Example usage:
    >>> import torch.nn as nn
    >>> import torch
    >>> class TestModel(nn.Module):
    ...    def __init__(self):
    ...        super().__init__()
    ...        self.fc = nn.Linear(in_features=1000, out_features=10)
    ...        self.conv = nn.Conv2d(
    ...            in_channels=3, out_channels=10, kernel_size=1
    ...        )
    ...        self.act = nn.ReLU()
    ...    def forward(self, x):
    ...        return self.fc(self.act(self.conv(x)).flatten(1))
    >>> model = TestModel()
    >>> inputs = (torch.randn((1,3,10,10)),)
    >>> maddss = maddsCountAnalysis(model, inputs)
    >>> maddss.total()
    13000
    >>> maddss.total("fc")
    10000
    >>> maddss.by_operator()
    Counter({"addmm" : 10000, "conv" : 3000})
    >>> maddss.by_module()
    Counter({"" : 13000, "fc" : 10000, "conv" : 3000, "act" : 0})
    >>> maddss.by_module_and_operator()
    {"" : Counter({"addmm" : 10000, "conv" : 3000}),
     "fc" : Counter({"addmm" : 10000}),
     "conv" : Counter({"conv" : 3000}),
     "act" : Counter()
    }
    """

    def __init__(
        self,
        model: nn.Module,
        inputs: Union[Tensor, Tuple[Tensor, ...]],
        SUPPORTED_OPS: Dict[str, Handle],
    ) -> None:
        super().__init__(model=model, inputs=inputs)
        self.set_op_handle(**SUPPORTED_OPS)

    __init__.__doc__ = JitModelAnalysis.__init__.__doc__


def compute_madds(
    model: nn.Module,
    sizes: Tuple[int],
    ignore_norm: bool = True,
    ignore_elementwise: bool = True,
    supported_ops: Optional[Dict[str, Handle]] = None,
) -> Tuple[DefaultDict[str, float], Counter[str]]:
    """
    Given a model and an input to the model, compute the per-operator Gflops
    of the given model.
    Args:
        model (nn.Module): The model to compute flop counts.
        inputs (tuple): Inputs that are passed to `model` to count flops.
            Inputs need to be in a tuple.
        supported_ops (dict(str,Callable) or None) : provide additional
            handlers for extra ops, or overwrite the existing handlers for
            convolution and matmul and einsum. The key is operator name and the value
            is a function that takes (inputs, outputs) of the op. We count
            one Multiply-Add as one FLOP.
    Returns:
        tuple[defaultdict, Counter]: A dictionary that records the number of
            gflops for each operation and a Counter that records the number of
            unsupported operations.
    """
    ops = copy.deepcopy(DENSE_OPS)
    if not ignore_norm:
        ops.update(NORM_OPS)
    if not ignore_elementwise:
        ops.update(ELEMENTWISE_OPS)
    if supported_ops is not None:
        ops.update(supported_ops)

    return MaddsAnalysis(model, torch.rand(sizes), ops).total()
