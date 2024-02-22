from dataclasses import dataclass
import torch
from torch import nn
import abc
from collections import namedtuple

@dataclass
class Camera:
    """
    Simple dataclass to hold properties used to project Gaussians
    """
    # Intrinsics
    fx:float
    fy:float
    cx:float
    cy:float
    projmat:torch.Tensor

    # Extrinsics
    viewmat:torch.Tensor

    # Misc
    H:int
    W:int
    znear:float=0.01
    zfar:float=100.0


    def project_points(
        self,
        points:torch.Tensor,
        res_agnostic:bool=True,
        remove_invisible:bool=True
    ) -> torch.Tensor:
        """
        Project a tensor of points to the image plane

        TODO: finish this

        Parameters:
        - `points:torch.Tensor` of shape `N,3`

        Returns:
        - `pixels:torch.Tensor` of shape `M,3` of XYZ pixel coordinates
        """
        # 3D XYZ to 4D homogeneous
        points_hom = torch.hstack((points, torch.ones(points.shape[0], device=points.device)))

        # Project to pixel coordinates
        pixels_hom = points_hom @ self.viewmat.T @ self.projmat.T

        # Back to 3D XYZ
        pixels = pixels_hom[:,:3] / pixels_hom[:,3]

        # Remove points outside view frustum
        if remove_invisible:
            mask = (
                  ( pixels[:,0]  < 0 )
                & ( pixels[:,0] >= self.W )
                & ( pixels[:,1]  < 0 )
                & ( pixels[:,1] >= self.H )
                & ( pixels[:,2]  < self.znear )
                & ( pixels[:,2] >= self.zfar )
            )

            pixels = pixels[mask]

        # Rescale to be resolution agnostic
        if res_agnostic:
            pixels = pixels / max(self.H, self.W)

        return pixels


CameraParameterization = namedtuple(
    "CameraParameterization",
    ['camera','flat_residuals','unflatten_fn','aux']
)
class CameraResidual(abc.ABC):
    """
    Abstract class for camera residual transformations
    """

    def create_residuals(self, camera:Camera) -> CameraParameterization:
        """
        Initialize residuals from Camera instance
        """
        raise NotImplementedError()

    def transform(self, camera:Camera, residuals:dict[str,torch.Tensor]) -> Camera:
        """
        Get transformed Camera instance from base Camera and unflattened residuals
        """
        raise NotImplementedError()

    def get_precondition_matrix(self, parameterization:CameraParameterization) -> torch.Tensor:

        def fwd(flat_residuals):
            residuals = parameterization.unflatten_fn(flat_residuals)

            camera = self.transform(parameterization.camera, residuals)

            points = torch.rand((1000,3))*2 - 1

            pixels = camera.project_points(points)

            return torch.linalg.norm(pixels, axis=1)

        jac = torch.autograd.functional.jacobian(fwd,parameterization.flat_residuals)
        sigma = jac.T @ jac
        print(sigma.shape)
        # TODO convert to sqrt_inv
