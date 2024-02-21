import torch
from torch import nn

from .camera import Camera
from src.utils import qvec2rotmat

class OptimizableCamera(Camera, nn.Module):
    use_residual = True
    use_precondition = True

    def __init__(self, *args, **kwargs):
        super(Camera, self).__init__()
        super().__init__(*args, **kwargs)

        # Parameters:
        self._residuals = {
            "quat": nn.Parameter(torch.tensor([1,0,0,0], dtype=torch.float32, device=self.device), requires_grad=True),
            "trans": nn.Parameter(torch.zeros(3, dtype=torch.float32, device=self.device), requires_grad=True),
            "fx": nn.Parameter(torch.zeros(1, dtype=torch.float32, device=self.device), requires_grad=True),
            "fy": nn.Parameter(torch.zeros(1, dtype=torch.float32, device=self.device), requires_grad=True),
        }

    @property
    def residuals(self) -> dict[str, nn.Parameter]:
        residuals = self._residuals

        if self.use_residual:
            sections = [param.shape[0] for param in residuals.values()]

            vec = torch.concat(residuals.values())

            # TODO add precondition matrix here here
            vec_prec = vec @ torch.eye(vec.shape[0], device=self.device)

            residuals = torch.split(vec_prec, sections)

        return residuals

    @property
    def viewmat(self) -> torch.Tensor:
        viewmat = super().viewmat

        residual = torch.vstack((
            torch.hstack((qvec2rotmat(self.residual_quat),self.residual_trans.unsqueeze(1))),
            torch.tensor([[0,0,0,1]], device=self.device)
        ))

        return residual @ viewmat

    @property
    def fx(self):
        fx = super().fx
        return fx + self.residual_fx

    @property
    def fy(self):
        fy = super().fy
        return fy + self.residual_fy


    def get_precondition_matrix(self) -> torch.Tensor:
        # Sample some random points
        N=1000
        pts = torch.randn(N,3, device=self.device)

        # Get jacobian of projecting these points
        jac = torch.autograd.functional.jacobian(self.project_points, pts, create_graph=False)

        # Sigma = J^T @ J
        jtj = jac.T @ jac

        # P = Sigma^{-1/2}
        eigval, eigvec = torch.linalg.eigh(jtj)
        prec = eigvec * torch.sqrt(eigval) @ torch.linalg.inv(eigvec)

        return prec
