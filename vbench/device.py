"""Device selection shared by the public API and evaluation launchers."""

import os
import torch

from .distributed import dist_init, get_world_size


def resolve_device(device):
    """Normalize an API device and initialize optional single-process NPU support."""
    name = str(device)
    if name.split(":", 1)[0] == "npu":
        if int(os.environ.get("WORLD_SIZE", "1")) != 1 or get_world_size() != 1:
            raise RuntimeError("Ascend evaluation currently supports a single NPU only")
        # Register the device type before constructing torch.device.
        import torch_npu
        if not torch.npu.is_available():
            raise RuntimeError("The requested Ascend NPU is not available")
        device = torch.device("npu:0" if name == "npu" else name)
        torch.npu.set_device(device)
        return device
    return torch.device(device)


def initialize_device(device):
    """Preserve CUDA launcher initialization; NPU uses one process without NCCL."""
    name = str(device)
    if name != "cuda" and name.split(":", 1)[0] != "npu":
        raise ValueError(f"Unsupported evaluation device: {device}; use cuda or npu:<index>")
    device = resolve_device(device)
    if device.type == "cuda":
        dist_init()
    return device
