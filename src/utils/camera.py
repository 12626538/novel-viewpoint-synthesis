import torch
import math
import numpy as np

class Camera:
    uid=0
    def __init__(
        self,
        gt_image:torch.Tensor, # Ground truth, shape HxWxC
        R:np.ndarray, # World2View rotation matrix, shape 3x3
        t:np.ndarray, # World2View translation vector, shape 3
        device:torch.device,
        fovx:float,
        fovy:float,
        znear:float=0.01,
        zfar:float=100.,
        name:str=None
    ):
        if name is None:
            name = f"cam{Camera.uid:05}.png"
            Camera.uid+=1
        self.name=name

        self.R = R
        self.t = t

        self.znear = znear
        self.zfar = zfar

        self.gt_image = gt_image.to(device)
        self.H,self.W = gt_image.shape[:2]

        self.fovx = fovx
        self.fovy = fovy

        self.device = device

    @property
    def fx(self) -> float:
        return self.W / (2 * math.tan(self.fovx / 2))

    @property
    def fy(self) -> float:
        return self.H / (2 * math.tan(self.fovy / 2))

    @property
    def viewmat(self) -> torch.Tensor:
        V = np.eye(4)

        V[:3,:3] = self.R
        V[3,:3] = self.t

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
        return torch.tensor(
            [[2 * n / (r - l), 0.0, (r + l) / (r - l), 0.0],
            [0.0, 2 * n / (t - b), (t + b) / (t - b), 0.0],
            [0.0, 0.0, (f + n) / (f - n), -1.0 * f * n / (f - n)],
            [0.0, 0.0, 1.0, 0.0]],
            device=self.device,
        )

    def __str__(self):
        return f"Camera(name='{self.name}')"
