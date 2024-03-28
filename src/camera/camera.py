import torch
import math
import numpy as np

from src.utils import fov2focal,get_projmat,get_viewmat

class Camera(object):
    """
    Base class for cameras, holds simple information about pose and camera
    intrinsics. Optionally also has a Tensor as `gt_image` property for
    training.
    """
    uid=0
    def __init__(
            self,
            # INTRINSICS
            fovx:float=1.5708,fovy:float=1.0472,
            cx_frac:float=.5, cy_frac:float=.5,
            zsign:float=1,
            # EXTRINSICS
            R:np.ndarray=np.eye(3),
            t:np.ndarray=np.zeros(3),
            #MISC
            gt_image:torch.Tensor=None,
            H:int=1080,W:int=1920,
            znear:float=0.01, zfar:float=100.,
            name:str=None,
            device='cuda',
        ):
        """
        # Create Camera instance
        This is the base class used to project Gaussians.

        All arguments are defined in a resolution-agnostic way
        such that upscaling `gt_image` can be done without having to
        recompute `fx,fy,cx,cy`.

        ## Parameters

        ### Intrinsics
        Parameters to set up the projection matrix
        - `fovx:float`,`fovy:float` - Field-Of-View in X and Y direction.
            defaults to 90 and 60 degrees in x and y direction
        - `cx_frac:float`, `cy_frac:float` - Fraction of principle points
            in X and Y direction to the scale, such that `cx = cx_frac * W`
            where `cx` is principle point in X direction and W is image width.
            Default is `0.5` (principle point is image center).

        ### Extrinsics
        Parameters to set up the `4x4` World-To-Camera matrix
        - `R:np.ndarray` - `3x3` rotation matrix
        - `t:np.ndarray` - translation vector of size `3,`

        ### Misc
        - `gt_image:Optional[torch.Tensor]` - Optional; ground truth image,
            if unset, specify `H,W` instead
        - `H:int`,`W:int` - Height and width of generated view,
            overwritten if `gt_image` is set. Defaults to fullHD (`1920x1080`)
        - `znear:float`,`zfar:float` - Distance of near and far plane,
            defaults to `0.01, 100`
        - `name:str` - Unique identifier of camera, defaults to `cam{uid:05d}.png`
            for a unique, incrementing id
        - `device='cuda'` - What device to put all tensors to, default is CUDA.
        """
        super().__init__()

        # Default name
        if name is None:
            name = f"cam{Camera.uid:05}.png"
            Camera.uid+=1
        self.name=name

        # Save device
        self.device = device

        # Set Ground Truth image
        if gt_image is not None:
            self.set_gt_image(gt_image)

        elif H is not None and W is not None:
            self.gt_image = None
            self.H, self.W = H,W

        else:
            raise ValueError("Either specify gt_image or H,W parameters "
                "when initializing Camera instance, got neither")

        # Intrinsics
        self.znear = znear
        self.zfar = zfar
        self.zsign = zsign

        self.fx = fov2focal(fovx, self.W)
        self.fy = fov2focal(fovy, self.H)

        self.cx = cx_frac * self.W
        self.cy = cy_frac * self.H

        self.viewmat = torch.from_numpy( get_viewmat(
            R=R, t=t
        ) ).to(device=self.device, dtype=torch.float32)

        self.projmat = torch.from_numpy( get_projmat(
            znear=znear, zfar=zfar,
            fovx=fovx, fovy=fovy,
            zsign=zsign,
        ) ).to(device=self.device, dtype=torch.float32)

        self.R = R
        self.t = t
        self.loc = -t @ R

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
        if self.gt_image is not None:
            self.gt_image = self.gt_image.to(device)
        self.viewmat = self.viewmat.to(device)
        self.projmat = self.projmat.to(device)

        return self


    def __str__(self):
        return (
            "Camera instance\n"
            f"\tname: '{self.name}'\n"
            f"\tDevice: {self.device}\n"
            f"\tH,W: {self.H,self.W}\n"
            f"\tcx,cy: {self.cx:.2f}, {self.cy:.2f}\n"
            f"\tfx,fy: {self.fx:.2f}, {self.fy:2f}\n"
            f"\tHas gt image? {self.gt_image is not None}\n"
            f"\tviewmat:\n{self.viewmat}\n"
            f"\tprojmat:\n{self.projmat}\n"
        )
