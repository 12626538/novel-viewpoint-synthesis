import torch
import math
import numpy as np

from src.utils import fov2focal,get_projmat,get_viewmat

class Camera:
    uid=0
    def __init__(
        self,
        R:np.ndarray, # World2View rotation matrix, shape 3x3
        t:np.ndarray, # World2View translation vector, shape 3
        device:torch.device,
        fovx:float,
        fovy:float,
        gt_image:torch.Tensor=None, # Ground truth, shape CxHxW
        H:int=None,W:int=None, # Height and Width, specify if gt_image is unspecified
        znear:float=0.01,
        zfar:float=100.,
        name:str=None
    ):
        """
        - `R:np.ndarray` - 3x3 rotation matrix
        - `t:np.ndarray` - translation vector of size 3,
        - `device` - What device to use
        - `fovx:float`,`fovy:float` - Field-Of-View in X and Y direction
        - `gt_image:Optional[torch.Tensor]` - Optional ground truth image, if unset, specify H,W instead
        - `H:int`,`W:int` - Height and width of generated view, overwritten if `gt_image` is set
        - `znear:float`,`zfar:float` - Distance of near and far plane, used for projection matrix
        - `name:str` - Unique identifier of camera, will use automatic incremental ID if unspecified
        """
        if name is None:
            name = f"cam{Camera.uid:05}.png"
            Camera.uid+=1
        self.name=name

        self.device = device

        self.R = R
        self.t = t

        self.znear = znear
        self.zfar = zfar

        self.fovx = fovx
        self.fovy = fovy

        # Set Ground Truth image
        if gt_image is not None:
            self.set_gt_image(gt_image)
        elif H is not None and W is not None:
            self.gt_image = None
            self.H, self.W = H,W
        else:
            raise ValueError("Either specify gt_image or H,W parameters when initializing Camera instance, got neither")

        # pre-compute useful features for rendering
        self.projmat = torch.from_numpy( get_projmat(
            znear=znear,zfar=zfar,
            fovx=fovx,fovy=fovy
        ) ).to(device=device, dtype=torch.float32)

        self.viewmat = torch.from_numpy( get_viewmat(
            R=R, t=t
        ) ).to(device=device, dtype=torch.float32)

        self.fx = fov2focal(self.fovx, self.W)
        self.fy = fov2focal(self.fovy, self.H)

        # World coordinate location
        self.loc = -t @ R


    def set_gt_image(self, image:torch.Tensor):
        """
        Centralized method to set `Camera.gt_image`, `Camera.H` and `Camera.W`
        """
        self.gt_image = image.to(self.device)
        self.H,self.W = image.shape[-2:]


    def to(self, device) -> 'Camera':
        """
        Set device for all tensors
        """
        self.device = device
        self.gt_image = self.gt_image.to(device)
        self.viewmat = self.viewmat.to(device)
        self.projmat = self.projmat.to(device)
        return self


    def __str__(self):
        return f"Camera(name='{self.name}')"
