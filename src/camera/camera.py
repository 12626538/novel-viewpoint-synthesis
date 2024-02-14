import torch
import math
import numpy as np

from src.utils import fov2focal,get_projmat,get_viewmat

class Camera(object):
    """
    # Base class for cameras
    """
    uid=0
    def __init__(
            self,
            # INTRINSICS
            fovx:float=1.5708,fovy:float=1.0472,
            H:int=1920,W:int=1080,
            znear:float=0.01, zfar:float=100.,
            # EXTRINSICS
            R:np.ndarray=np.eye(3),
            t:np.ndarray=np.zeros(3),
            #MISC
            gt_image:torch.Tensor=None,
            name:str=None,
            device='cuda',
        ):
        """
        # Create Camera instance
        This is the base class used to project Gaussians. Comporable to the
        JaxCam class used for ZipNeRF.

        ## Parameters

        ### Intrinsics
        Parameters to set up the projection matrix
        - `fovx:float`,`fovy:float` - Field-Of-View in X and Y direction.
            defaults to 90 and 60 degrees in x and y direction
        - `znear:float`,`zfar:float` - Distance of near and far plane,
            defaults to `0.01, 100`

        ### Extrinsics
        Parameters to set up the `4x4` World-To-Camera matrix
        - `R:np.ndarray` - `3x3` rotation matrix
        - `t:np.ndarray` - translation vector of size `3,`

        ### Misc
        - `device='cuda'` - What device to put all tensors to, default is CUDA
        - `H:int`,`W:int` - Height and width of generated view,
            overwritten if `gt_image` is set. Defaults to fullHD (`1920x1080`)
        - `gt_image:Optional[torch.Tensor]` - Optional; ground truth image,
            if unset, specify `H,W` instead
        - `name:str` - Unique identifier of camera, defaults to `cam{uid:05d}.png`
            for a unique, incrementing id
        """
        super().__init__()

        # Default name
        if name is None:
            name = f"cam{Camera.uid:05}.png"
            Camera.uid+=1
        self.name=name

        # Save device
        self.device = device

        # Intrinsics
        self.znear = znear
        self.zfar = zfar

        self.fovx = fovx
        self.fovy = fovy

        # Extrinsics
        self.R = R
        self.t = t

        # Set Ground Truth image
        if gt_image is not None:
            self.set_gt_image(gt_image)

        elif H is not None and W is not None:
            self.gt_image = None
            self.H, self.W = H,W

        else:
            raise ValueError("Either specify gt_image or H,W parameters when initializing Camera instance, got neither")


    @property
    def loc(self) -> np.ndarray:
        """Camera location in world coordinates"""
        return -self.t @ self.R


    @property
    def projmat(self) -> torch.Tensor:
        """4x4 Projection matrix, moving world coordinates to pixel coordinates"""
        return torch.from_numpy( get_projmat(
            znear=self.znear,zfar=self.zfar,
            fovx=self.fovx,fovy=self.fovy
        ) ).to(device=self.device, dtype=torch.float32)


    @property
    def viewmat(self) -> torch.Tensor:
        """
        World-To-View matrix, a 4x4 rigid body transform moving points in world
        coordinates to camera coordinates
        """
        return torch.from_numpy( get_viewmat(
            R=self.R, t=self.t
        ) ).to(device=self.device, dtype=torch.float32)


    @property
    def fx(self) -> float:
        """Focal in X direction"""
        return fov2focal(self.fovx, self.W)


    @property
    def fy(self) -> float:
        """Focal in Y direction"""
        return fov2focal(self.fovy, self.H)


    def set_gt_image(self, image:torch.Tensor):
        """
        Centralized method to set `gt_image, H, W`
        """
        self.gt_image = image.to(self.device)
        self.H,self.W = image.shape[-2:]


    def to(self, device) -> 'Camera':
        """
        Set device for all relevant tensors
        """
        self.device = device
        self.gt_image = self.gt_image.to(device)

        return self


    def __str__(self):
        return (
            "Camera\n"
            f"\tname='{self.name}'\n"
            f"\tviewmat='{self.viewmat}'\n"
            f"\tprojmat='{self.projmat}'\n"
            f"\tH,W='{self.H,self.W}'\n"
        )
