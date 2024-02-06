import torch
import math
import numpy as np

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

        if gt_image is not None:
            self.set_gt_image(gt_image)
        elif H is not None and W is not None:
            self.gt_image = None
            self.H, self.W = H,W
        else:
            raise ValueError("Either specify gt_image or H,W parameters when initializing Camera instance, got neither")



    def set_gt_image(self, image:torch.Tensor):
        """
        Centralized method to set `Camera.gt_image`, `Camera.H` and `Camera.W`
        """
        self.gt_image = image.to(self.device)
        self.H,self.W = image.shape[-2:]

    # Use @properties to not have to worry about updating things when
    # Camera.W changes etc
    @property
    def fx(self) -> float:
        """Focal distance in x direction"""
        return self.W / (2 * math.tan(self.fovx / 2))

    @property
    def fy(self) -> float:
        """Focal distance in y direction"""
        return self.H / (2 * math.tan(self.fovy / 2))

    @property
    def viewmat(self) -> torch.Tensor:
        V = np.eye(4)

        V[:3,:3] = self.R.T
        V[:3,3] = self.t

        return torch.tensor(V, dtype=torch.float, device=self.device)

    @property
    def projmat(self) -> torch.Tensor:
        """
        https://github.com/nerfstudio-project/nerfstudio/blob/9e33b437dff6df5a9579c04b1eba46640df88a96/nerfstudio/models/gaussian_splatting.py#L73
        """
        t = self.znear * math.tan(0.5 * self.fovy)
        b = -t
        r = self.znear * math.tan(0.5 * self.fovx)
        l = -r
        n = self.znear
        f = self.zfar

        zsign = 1.0
        return torch.tensor(
            [[2 * n / (r - l), 0.0, (r + l) / (r - l), 0.0],
            [0.0, 2 * n / (t - b), (t + b) / (t - b), 0.0],
            [0.0, 0.0, (f + n) / (f - n), -1.0 * zsign * f * n / (f - n)],
            [0.0, 0.0, zsign, 0.0]],
            device=self.device,
        )

    def to(self, device) -> 'Camera':
        self.device = device
        self.gt_image = self.gt_image.to(device)
        return self

    def __str__(self):
        return f"Camera(name='{self.name}')"
