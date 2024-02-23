import torch
import torch.nn.functional as F
import torch.optim.lr_scheduler as scheduler

import numpy as np

def clip(value, a, b=None):
    """
    Clip value to interval `[a,b]`

    If `b` is not set, clip to `[0, a]`
    """
    if b is None:
        b=a
        a=0
    return min(max(value, a), b)

def cosine_warmup(iters, start=1e-8, end=1e-0):
    """
    Creates a cosine decay warmup schedule that, given
    an iteration `iter`, returns at `start` on `iter=0`
    and `end` on `iter>=iters`
    """

    def warmup(iter):
        # Starts at 1, ends at 0
        alpha = 0.5 * (1 + np.cos(np.pi * clip(iter, iters) / iters))

        # Interpolate start and end
        return alpha * start + (1-alpha) * end

    return warmup

def log_linear(iters, start=1e-0, end=1e-2):
    """
    Log-Linearly interpolate between start and end
    """

    log_start = np.log(start)
    log_end = np.log(end)

    def log_linear_interp(iter):
        frac = clip(iter, iters) / iters

        return np.exp( (1. - frac) * log_start + frac * log_end )

    return log_linear_interp
