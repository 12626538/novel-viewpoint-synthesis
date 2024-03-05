from typing import Callable
import torch
from torch import nn
from torch.nn import L1Loss, MSELoss, functional as F
from torch import Tensor
import lpips
"""
Below is adapted from
https://github.com/graphdeco-inria/gaussian-splatting/blob/main/utils/loss_utils.py

Copyright (C) 2023, Inria
GRAPHDECO research group, https://team.inria.fr/graphdeco
All rights reserved.

This software is free for non-commercial, research and evaluation use
under the terms of the LICENSE.md file.

For inquiries contact  george.drettakis@inria.fr
"""
def gaussian(size, sigma):
    """
    From
    https://github.com/VainF/pytorch-msssim/blob/b057b072dd869ae3f6b88543786f44d008315f69/pytorch_msssim/ssim.py#L11

    Creates 1D Gaussian filter of given size and variance sigma
    """
    coords = torch.arange(size, dtype=torch.float)
    coords -= size // 2

    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g /= g.sum()

    return g


def create_window(window_size, channel) -> torch.Tensor:
    """
    Create Gaussian kernel of shape `channel,1,window_size,window_size`
    """
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    return window.requires_grad_(False)


class SSIM(torch.nn.Module):
    def __init__(
        self,
        num_channel:int=3,
        window_size:int=11,
        size_average:bool=True,
        device='cuda',
    ) -> None:
        super().__init__()
        self.win_size = window_size
        self.size_average = size_average
        self.channel = num_channel
        self.win = create_window(window_size=window_size, channel=num_channel).to(device)

    def forward(self, X:Tensor, Y:Tensor) -> Tensor:
        win = create_window(window_size=self.win_size, channel=self.channel).cuda()

        mux = F.conv2d(X, win, padding=self.win_size // 2, groups=self.channel)
        muy = F.conv2d(Y, win, padding=self.win_size // 2, groups=self.channel)

        mux_sq = mux.pow(2)
        muy_sq = muy.pow(2)
        mux_muy = mux * muy

        sigmaxx = F.conv2d(X * X, win, padding=self.win_size // 2, groups=self.channel) - mux_sq
        sigmayy = F.conv2d(Y * Y, win, padding=self.win_size // 2, groups=self.channel) - muy_sq
        sigmaxy = F.conv2d(X * Y, win, padding=self.win_size // 2, groups=self.channel) - mux_muy

        C1 = 0.01 ** 2
        C2 = 0.03 ** 2

        ssim_map = ((2 * mux_muy + C1) * (2 * sigmaxy + C2)) / ((mux_sq + muy_sq + C1) * (sigmaxx + sigmayy + C2))

        if self.size_average:
            return ssim_map.mean()
        else:
            return ssim_map.mean(1).mean(1).mean(1)

class DSSIM(SSIM):
    def forward(self, X:Tensor, Y:Tensor) -> Tensor:
        return 1. - super().forward(X,Y)

# </Adapted From>

class LPIPS(nn.Module):
    """
    LPIPS wrapper to convert datarange [0,1] to [-1,1] and apply LPIPS
    """
    def __init__(self, *args, device='cuda', **kwargs):
        super().__init__()
        # Initialize LPIPS model
        self.lpips = lpips.LPIPS(*args, verbose=False, **kwargs)
        # Set device
        self.lpips.to(device)

    def forward(self, in0, in1, retPerLayer=False, normalize=False):
        # Convert data range [0,1] to [-1,1]
        in0 = in0 * 2. - 1.
        in1 = in1 * 2. - 1.

        # Call LPIPS method
        return self.lpips(in0, in1, retPerLayer, normalize)


LOSSES:dict[str,type[nn.Module]] = {
    'mae': L1Loss,
    'mse': MSELoss,
    'lpips': LPIPS,
    'dssim': DSSIM,
}
loss_fn = Callable[[torch.Tensor,torch.Tensor], tuple[torch.Tensor, dict[str,float]]]
def get_loss_fn(args, module_kwargs={}) -> loss_fn:
    """
    Get loss function

    Initializes multiple Loss modules and returns a callable that
    takes as input two tensors and returns a tuple
    - the loss (Tensor), a weighted sum of the loss modules
    - a dict mapping the name of each type of loss to the unweighted value

    Current losses:
    - `'mae': L1Loss` Mean Absolute Error
    - `'mse': MSELoss` Mean Squared Error
    - `'lpips': LPIPS` Perceptual loss
    - `'dssim': DSSIM` Discounted Structual Similarity Index Measure

    Args:
    - `args:dict` A dictionary with optionally keys 'loss_weight_[NAME]' (where
        NAME is the name of a loss module) and weights as values. If a key does
        not exist, set weight to zero
    - `module_kwargs:optional[dict]` Additional arguments passed to each of the
        loss modules on initialization.

    Returns
    - `loss_fn` A loss function, see doc below
    """
    # Initialize loss modules
    losses = {
        name: (LOSSES[name](**module_kwargs), args.get(f'loss_weight_{name}', 0.))
        for name in LOSSES
    }

    # Define loss function
    def loss_fn(inpA, inpB):
        """
        Given two images `inpA,inpB`, call each loss function and record their
        loss. Weigh each loss based on the weights defined by `losses` and compute
        their sum. Also keep a dictionary of each unweighted loss.

        Note that the values in the dictionary are regular floats, and therefore
        cannot be used to backpropagate.

        Args:
        - `inpA:Tensor, inpB:Tensor` Tensors of shape `C,H,W` with values in
            the range `[0,1]`

        Returns:
        - `loss_combined:Tensor` A 0-dimensional Tensor with the weighted sum of
            the losses
        - `individual_loss:dict[str,float]` A dictionary mapping the name of
            each loss to the unweighted value
        """
        # Set up combined and individual loss
        loss_combined = 0.
        individual_losses = {}

        for (name, (func, weight)) in losses.items():

            # Compute loss
            loss = func(inpA, inpB).squeeze()

            # Record loss
            individual_losses[name] = loss.item()
            loss_combined += loss * weight

        return loss_combined, individual_losses
    return loss_fn
