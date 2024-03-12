

from dataclasses import dataclass,replace
import torch
from torch import nn
from typing import Callable,Any

def get_projmat(
        fx:float,fy:float,
        H:int,W:int,
        n:float,f:float
    ) -> torch.Tensor:
    """
    From http://www.songho.ca/opengl/gl_projectionmatrix.html

    Assumes r=-l, t=-b (principle point is in the center of the view frustum)

    This because `gsplat.project_gaussians` already accounts for non-centered
    principle point.

    Diffirientable w.r.t all input arguments
    """

    projmat = torch.zeros(4,4,device=fx.device)

    sign = -1
    projmat[0,0] = 2 * fx / W
    projmat[1,1] = 2 * fy / H
    projmat[2,2] = (f+n) / (n-f) * sign
    projmat[2,3] = 2*n*f / (n-f)
    projmat[3,2] = -sign

    return projmat


def inv_sqrtm(mat:torch.Tensor, eps=torch.finfo(torch.float32).tiny) -> torch.Tensor:
    """
    From https://github.com/jonbarron/camp_zipnerf/blob/16206bd88f37d5c727976557abfbd9b4fa28bbe1/internal/spin_math.py#L89
    """

    # Computing diagonalization
    eigvals, eigvecs = torch.linalg.eigh(mat)

    # eigvals += 1e-10

    # Inv Sqrt of eigenvalues, but prevent NaNs on unstable values
    eigvals = torch.where(eigvals > eps, eigvals, torch.finfo(torch.float32).eps)
    scaling = (1. / torch.sqrt(eigvals)).view(1,-1)

    # Reconstruct matrix
    return (eigvecs * scaling) @ eigvecs.moveaxis(-2, -1)

@dataclass
class DCCamera:
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
    gt_image:torch.Tensor
    znear:float=0.01
    zfar:float=100.0

    @property
    def loc(self):
        return self.world_position().detach().cpu().numpy()


    def world_position(self)->torch.Tensor:
        """Get Camera position in world coordinates"""
        R = self.viewmat[:3,:3]
        t = self.viewmat[:3,3]
        return -t @ R


    def project_points(
        self,
        points:torch.Tensor,
        res_agnostic:bool=False,
        remove_invisible:bool=True,
    ) -> torch.Tensor:
        """
        Project a tensor of points to the image plane

        Parameters:
        - `points:torch.Tensor` of shape `N,3`

        Returns:
        - `pixels:torch.Tensor` of shape `M,2` of XY pixel coordinates where
            `M` is the number of points inside the view frustum if
            `remove_invisible` is set to True, and `M=N` otherwise
        """
        # 3D XYZ to 4D homogeneous
        points_hom = torch.hstack((points, torch.ones(points.shape[0],1, device=points.device)))

        # Project to pixel coordinates
        # TODO: add externals too
        # pixels_hom = points_hom @ self.viewmat.T @ self.projmat.T
        pixels_hom = points_hom @ self.projmat.T

        # Back to XYZ
        pixels = pixels_hom[:,:-1] / pixels_hom[:,-1:]

        # Remove points outside view frustum
        if remove_invisible:
            mask = (
                  ( pixels[:,0] >= 0 )
                & ( pixels[:,0]  < self.W )
                & ( pixels[:,1] >= 0 )
                & ( pixels[:,1]  < self.H )
                # & ( pixels[:,2] >= self.znear )
                # & ( pixels[:,2]  < self.zfar )
            )

            pixels = pixels[mask]

        # Rescale to be resolution agnostic
        if res_agnostic:
            pixels = pixels / max(self.H, self.W)

        return pixels[:,:2]


def flatten(residuals:dict[str,torch.Tensor]):
    """
    Flatten a dictionary of tensors into a single tensor

    Also provides unflatten function, st
    >>> latent,unflatten_fn = flatten(residuals)
    >>> residuals == unflatten_fn(latent)
    holds
    """
    # Get names and sizes of the parameters
    names = list(residuals.keys())
    sections = [residuals[name].shape[0] for name in names]

    # Use names and sizes to set up unflatten function
    def unflatten_fn(latent:torch.Tensor) -> dict[str, torch.Tensor]:
        """Unflatten latent vector into residuals dictionary"""
        # Split based on sizes
        params = latent.split(sections)

        # Use names to set up residuals dictionary
        return { name:param for name,param in  zip(names, params) }

    # Actually flatten all the tensors
    latent = torch.concat([residuals[name] for name in names])

    return latent, unflatten_fn


@dataclass
class CameraParameterization:
    camera:DCCamera
    latent:nn.Parameter
    unflatten_fn:Callable[[nn.Parameter], dict[str, torch.Tensor]]
    aux:Any=None
    prec_mat:torch.Tensor=None

class CameraResidual():
    """
    Abstract class for camera residual transformations
    """
    use_residual:bool=True
    use_precondition:bool=False

    @staticmethod
    def create_residuals(camera:DCCamera) -> tuple[dict[str,torch.Tensor], Any]:
        """
        Initialize residuals from Camera instance
        """
        raise NotImplementedError()


    @staticmethod
    def transform(camera:DCCamera, residuals:dict[str,torch.Tensor], aux:Any) -> DCCamera:
        """
        Get transformed Camera instance from base Camera and unflattened residuals
        """
        raise NotImplementedError()


    @classmethod
    def parameterize(cls, camera:DCCamera) -> CameraParameterization:
        residuals, aux = cls.create_residuals(camera)
        latent, unflatten_fn = flatten(residuals)

        param = CameraParameterization(camera, latent, unflatten_fn,aux)

        if cls.use_precondition:
            param.prec_mat = cls.get_precondition_matrix(param)

        return param


    @classmethod
    def get_camera(cls, param:CameraParameterization) -> DCCamera:
        """
        Given parameterization, get transformed Camera instance

        Can be thought of as the inverse of `parameterize`
        """
        residuals = cls.get_residuals(param)
        return cls.transform(param.camera, residuals, param.aux)

    @classmethod
    def get_residuals(cls, param:CameraParameterization) -> dict[str,torch.Tensor]:
        """
        Get residuals as a dict from parameterization instance

        Also applies any processing of latent-to-residuals
        """

        latent = param.latent

        if cls.use_precondition:
            latent = latent @ param.prec_mat

        return param.unflatten_fn(latent)


    @classmethod
    def get_precondition_matrix(cls, param:CameraParameterization) -> torch.Tensor:
        """
        Get precondition matrix from parameterization
        """

        # Wrap the Camera.project_points in a function to accept a tensor as arg
        def fwd(latent):
            """
            Given a flattened residual vector, compute pixel locations
            of randomly generated points.
            """

            # Convert residuals to Camera instance
            residuals = param.unflatten_fn(latent)
            camera = cls.transform(param.camera, residuals, param.aux)

            # Generate random points in [0,100]^3
            points = torch.rand((1000,3), device=latent.device)*2.-1.

            # Project points to pixel coordinates
            pixels = camera.project_points(points)

            # print("Number of visible points:", pixels.shape[0])

            return pixels

        # Get jacobian, shape `M,2,K`
        jac = torch.autograd.functional.jacobian(fwd, param.latent, create_graph=False)

        jac = jac.mean(axis=1)

        # Convert to covariance, shape `K,K`
        sigma = jac.T @ jac

        # Convert to precondition matrix `P^-1 = Sigma^-1/2`
        return inv_sqrtm(sigma).detach().requires_grad_(False)


class UnnamedCameraType(CameraResidual):
    def create_residuals(camera:DCCamera) -> tuple[dict[str,torch.Tensor], Any]:
        device = camera.viewmat.device
        residuals = {
            "f": torch.zeros(1, device=device),
            "cx": torch.zeros(1, device=device),
            "cy": torch.zeros(1, device=device),
            "t": torch.zeros(3, device=device),
            "quat": torch.zeros(4, device=device),
        }

        aux = None

        return residuals, aux

    def transform(camera: DCCamera, residuals: dict[str, torch.Tensor], aux:Any) -> DCCamera:

        fx = camera.fx * torch.exp( residuals['f'] )
        fy = camera.fy * torch.exp( residuals['f'] )
        cx = camera.cx + residuals['cx']
        cy = camera.cy + residuals['cy']

        projmat = get_projmat(
            fx=fx,fy=fy,
            H=camera.H,W=camera.W,
            n=camera.znear,f=camera.zfar
        )

        return DCCamera(
            fx=fx, fy=fy, cx=cx, cy=cy,
            projmat=projmat, viewmat=camera.viewmat,
            H=camera.H, W=camera.W,
            gt_image=camera.gt_image, znear=camera.znear, zfar=camera.zfar
        )
