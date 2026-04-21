"""Device selection — picks the best available compute backend.

Priority: CUDA > MPS (Apple Silicon) > CPU.
Can be overridden via the 'requested' argument or pipeline config.
"""

from __future__ import annotations

import torch

from autolabel3d.utils.logging import get_logger

logger = get_logger(__name__)


def get_device(requested: str = "auto") -> torch.device:
    """Select the best available compute device.

    Args:
        requested: "auto" | "cpu" | "mps" | "cuda".

    Returns:
        torch.device instance.
    """
    if requested == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(requested)

    logger.info("Using device: %s", device)
    return device
