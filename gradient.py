import torch
from torch.utils.checkpoint import checkpoint

"""
PyTorch equivalent of memory-saving gradients.

Instead of rewriting gradients (like TensorFlow version),
we wrap parts of the forward pass with checkpoint().
"""

# =========================
# BASIC CHECKPOINT WRAPPER
# =========================

def checkpoint_function(fn, *args):
    """
    Wrap a function with torch checkpointing.
    """
    return checkpoint(fn, *args)


# =========================
# MODULE WRAPPER
# =========================

class CheckpointModule(torch.nn.Module):
    """
    Wrap any module to enable gradient checkpointing.
    """

    def __init__(self, module):
        super().__init__()
        self.module = module

    def forward(self, *inputs):
        def forward_fn(*inputs):
            return self.module(*inputs)

        return checkpoint(forward_fn, *inputs)


# =========================
# SEQUENTIAL CHECKPOINTING
# =========================

def checkpoint_sequential(functions, segments, input):
    """
    Split model into segments and checkpoint each.

    functions: list of layers/modules
    segments: number of chunks
    """
    return torch.utils.checkpoint.checkpoint_sequential(functions, segments, input)


# =========================
# OPTIONAL: APPLY TO MODEL
# =========================

def apply_gradient_checkpointing(model, every_n_layers=1):
    """
    Automatically wrap layers for checkpointing.

    every_n_layers:
        1 → checkpoint every layer
        2 → every 2 layers, etc.
    """

    layers = []

    for i, layer in enumerate(model.children()):
        if i % every_n_layers == 0:
            layers.append(CheckpointModule(layer))
        else:
            layers.append(layer)

    return torch.nn.Sequential(*layers)