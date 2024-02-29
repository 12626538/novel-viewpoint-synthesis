import lpips
import torch
from torch import nn

class LPIPS(nn.Module):
    # Make sure to only run it once
    lpips = None
    """
    LPIPS wrapper to convert datarange [0,1] to [-1,1] and apply LPIPS
    """
    def __init__(self, *args, device='cuda', **kwargs):
        super().__init__()
        # Initialize LPIPS model
        if self.lpips is None:
            self.lpips = lpips.LPIPS(*args, verbos=False, **kwargs)
            # Set device
            self.lpips.to(device)

    def forward(self, in0, in1, retPerLayer=False, normalize=False):
        # Convert data range [0,1] to [-1,1]
        in0 = in0 * 2. - 1.
        in1 = in1 * 2. - 1.

        # Call LPIPS method
        return self.lpips(in0, in1, retPerLayer, normalize)
