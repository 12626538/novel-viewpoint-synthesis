from dataclasses import dataclass,replace
import torch
from torch import nn
from typing import Callable,Any

def get_projmat(
        fx:float,fy:float,
        cx:float,cy:float,
        H:int,W:int,
        n:float,f:float
    ) -> torch.Tensor:
    """
    From https://stackoverflow.com/a/75355212

    Diffirientable w.r.t all input arguments
    """

    projmat = torch.zeros(4,4,device=fx.device)

    projmat[0,0] = 2 * fx / W
    projmat[0,2] = -(W - 2*cx)/W
    projmat[1,1] = 2 * fy / H
    projmat[1,2] = -(2*cy - H)/H
    projmat[2,2] = (n + f)/(n - f)
    projmat[2,3] = 2 * n * f / (n - f)
    projmat[3,2] = 1

    return projmat


def inv_sqrtm(mat:torch.Tensor, eps=torch.finfo(torch.float32).eps) -> torch.Tensor:
    """
    From https://github.com/jonbarron/camp_zipnerf/blob/16206bd88f37d5c727976557abfbd9b4fa28bbe1/internal/spin_math.py#L89
    """

    # Computing diagonalization
    eigvals, eigvecs = torch.linalg.eigh(mat)

    # Inv Sqrt of eigenvalues, but prevent NaNs on unstable values
    scaling = (1 / torch.sqrt(eigvals)).view(1,-1)
    scaling = torch.where(eigvals > eps, scaling, 0)

    # Reconstruct matrix
    return (eigvecs * scaling) @ eigvecs.T

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


    def world_position(self)->torch.Tensor:
        """Get Camera position in world coordinates"""
        R = self.viewmat[:3,:3]
        t = self.viewmat[:3,3]
        return -t @ R


    def project_points(
        self,
        points:torch.Tensor,
        res_agnostic:bool=True,
        remove_invisible:bool=False,
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
    names = list(residuals.keys())
    sections = [residuals[name].shape[0] for name in names]

    def unflatten_fn(flat_residuals:torch.Tensor) -> dict[str, torch.Tensor]:
        params = flat_residuals.split(sections)

        return { name:param for name,param in  zip(names, params) }

    flat_residuals = torch.concat([residuals[name] for name in names])

    return flat_residuals, unflatten_fn

@dataclass
class CameraParameterization:
    camera:Camera
    flat_residuals:nn.Parameter
    unflatten_fn:Callable[[nn.Parameter], dict[str, torch.Tensor]]
    aux:Any=None
    prec_mat:torch.Tensor=None

class CameraResidual():
    """
    Abstract class for camera residual transformations
    """
    use_residual:bool=True
    use_precondition:bool=True

    @staticmethod
    def create_residuals(camera:Camera) -> tuple[dict[str,torch.Tensor], Any]:
        """
        Initialize residuals from Camera instance
        """
        raise NotImplementedError()


    @staticmethod
    def transform(camera:Camera, residuals:dict[str,torch.Tensor]) -> Camera:
        """
        Get transformed Camera instance from base Camera and unflattened residuals
        """
        raise NotImplementedError()


    @classmethod
    def parameterize(cls, camera:Camera) -> CameraParameterization:
        residuals, aux = cls.create_residuals(camera)
        flat_residuals, unflatten_fn = flatten(residuals)

        param = CameraParameterization(camera, flat_residuals, unflatten_fn,aux)

        if cls.use_precondition:
            param.prec_mat = cls.get_precondition_matrix(param)

        return param


    @classmethod
    def get_camera(cls, param:CameraParameterization) -> Camera:
        residuals = cls.get_residuals(param)
        return cls.transform(param.camera, residuals)

    @classmethod
    def get_residuals(cls, param:CameraParameterization) -> dict[str,torch.Tensor]:
        """
        Get residuals as a dict from parameterization instance
        """

        flat_residuals = param.flat_residuals

        if cls.use_precondition:
            flat_residuals = flat_residuals @ param.prec_mat

        return param.unflatten_fn(flat_residuals)


    @classmethod
    def get_precondition_matrix(cls, parameterization:CameraParameterization) -> torch.Tensor:

        # Wrap the Camera.project_points in a function to accept a tensor as arg
        def fwd(flat_residuals):
            """
            Given a flattened residual vector, compute pixel locations
            of randomly generated points.
            """

            # Convert residuals to Camera instance
            residuals = parameterization.unflatten_fn(flat_residuals)
            camera = cls.transform(parameterization.camera, residuals)

            # Generate random points in [0,100]^3
            points = torch.rand((1000,3), device=flat_residuals.device)

            # Project points to pixel coordinates
            pixels = camera.project_points(points)

            return pixels.flatten()

        # Get jacobian, shape `2M,K`
        jac = torch.autograd.functional.jacobian(fwd, parameterization.flat_residuals, create_graph=False)

        # Convert to covariance, shape `K,K`
        sigma = jac.T @ jac

        # Convert to precondition matrix `P^-1 = Sigma^-1/2`
        return inv_sqrtm(sigma).detach().requires_grad_(False)


class CameraIntrinsics(CameraResidual):
    def create_residuals(camera:Camera) -> tuple[dict[str,torch.Tensor], Any]:
        device = camera.viewmat.device
        residuals = {
            "fx": torch.zeros(1, device=device),
            "fy": torch.zeros(1, device=device),
            "cx": torch.zeros(1, device=device),
            "cy": torch.zeros(1, device=device)
        }

        aux = None

        return residuals, aux

    def transform(camera: Camera, residuals: dict[str, torch.Tensor]) -> Camera:

        fx = camera.fx + residuals['fx']
        fy = camera.fy + residuals['fy']
        cx = camera.cx + residuals['cx']
        cy = camera.cx + residuals['cy']

        projmat = get_projmat(
            fx,fy,
            cx,cy,
            camera.H,camera.W,
            camera.znear,camera.zfar
        )

        return replace(
            camera,
            fx=fx,fy=fy,
            cx=cx,cy=cy,
            projmat=projmat
        )
