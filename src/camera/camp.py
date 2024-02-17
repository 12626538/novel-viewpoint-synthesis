import torch
from torch import nn

from .camera import Camera
from src.utils import get_projmat,qvec2rotmat

class OptimizableCamera(Camera, nn.Module):
    def __init__(self, *args, **kwargs):
        super(Camera, self).__init__()
        super().__init__(*args, **kwargs)

        # Parameters:
        self.residual_quat = nn.Parameter(torch.tensor([1,0,0,0], dtype=torch.float32, device=self.device), requires_grad=True)
        self.residual_trans = nn.Parameter(torch.zeros(3, dtype=torch.float32, device=self.device), requires_grad=True)

    @property
    def viewmat(self) -> torch.Tensor:
        viewmat = super().viewmat

        residual = torch.vstack((
            torch.hstack((qvec2rotmat(self.residual_quat),self.residual_trans.unsqueeze(1))),
            torch.tensor([[0,0,0,1]], device=self.device)
        ))

        return residual @ viewmat
