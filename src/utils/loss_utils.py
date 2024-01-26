import torch
from torch.nn import L1Loss, functional as F
from torch import Tensor
from torch.autograd import Variable

class CombinedLoss(torch.nn.Module):
    """
    Class to handle a weighted average of different loss functions

    Usage:
    >>> loss_func = CombinedLoss( L1Loss(), (DSSIMLoss(), 0.2) )
    >>> loss = loss_func(X, Y)

    Equivalent to
    >>> l1Loss = L1Loss()
    >>> dssimLoss = DSSIMLoss()
    >>> loss = 1* l1Loss(X,Y) + 0.2* dssimLoss(X, Y)
    """
    def __init__(self, *funcs) -> None:
        """

        Parameters:
        - `*funcs` - A sequence of either `nn.Module` or `tuple[nn.Module, float]`
        of loss functions. If no weight is specified, use a weight of 1
        """
        super().__init__()

        self.losses = []
        self.weights = []

        for func in funcs:
            if isinstance(func,tuple):
                func,weight = func
            else:
                weight = 1
            self.losses.append(func)
            self.weights.append(weight)

    def forward(self, X, Y):
        """
        Take weighted average of each loss func on input X,Y
        """
        loss = 0.

        for func,weight in zip(self.losses, self.weights):
            loss += weight * func(X,Y)

        return loss


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

class DSSIMLoss(torch.nn.Module):
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
        X = X.permute(2,0,1) # H,W,C -> C,H,W
        Y = Y.permute(2,0,1)

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
            return 1. - ssim_map.mean()
        else:
            return 1. - ssim_map.mean(1).mean(1).mean(1)
