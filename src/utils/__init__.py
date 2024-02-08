import torch
import numpy as np
import math
from PIL import Image
from torchvision import transforms

def sigmoid_inv(x):
    return torch.log(x/(1-x))

def qvec2rotmat(qvec:np.ndarray) -> np.ndarray:
    """
    Convert [w,x,y,z] quaternion to 3x3 rotation matrix
    """
    w,x,y,z = qvec

    R = np.zeros((3, 3))

    R[0, 0] = 1 - 2 * (y*y + z*z)
    R[0, 1] = 2 * (x*y - w*z)
    R[0, 2] = 2 * (x*z + w*y)
    R[1, 0] = 2 * (x*y + w*z)
    R[1, 1] = 1 - 2 * (x*x + z*z)
    R[1, 2] = 2 * (y*z - w*x)
    R[2, 0] = 2 * (x*z - w*y)
    R[2, 1] = 2 * (y*z + w*x)
    R[2, 2] = 1 - 2 * (x*x + y*y)
    return R


def focal2fov(focal:float, pixels:int) -> float:
    return 2*math.atan(pixels/(2*focal))

def fov2focal(fov:float, pixels:int) -> float:
    return pixels / (2 * math.tan(fov / 2))


def image_path_to_tensor(image_path:str, rescale:float=1) -> torch.Tensor:
    """
    Read image to tensor

    Rescale by a factor 1/rescale if specified

    Output shape is C,H,W
    """
    img = Image.open(image_path)

    if rescale != 1:
        img = img.resize((img.width//rescale, img.height//rescale))

    transform = transforms.ToTensor()

    img_tensor = transform(img)
    return img_tensor

def batch_qvec2rotmat(batch_qvec) -> torch.Tensor:
    """
    Convert a batch of Quaternion vectors to Rotation matrices

    Code from
    https://github.com/graphdeco-inria/gaussian-splatting/blob/2eee0e26d2d5fd00ec462df47752223952f6bf4e/utils/general_utils.py#L78

    Parameters:
    - `batch_qvec:torch.Tensor` of shape `N,4`

    Returns
    - `batch_rotmat:torch.Tensor` of shape `N,3,3`
    """

    # norm = torch.sqrt(batch_qvec[:,0]*r[:,0] + r[:,1]*r[:,1] + r[:,2]*r[:,2] + r[:,3]*r[:,3])
    q = batch_qvec / torch.linalg.norm(batch_qvec, dim=1,keepdim=True)

    R = torch.zeros((q.shape[0], 3, 3), device=q.device)

    w = q[:, 0]
    x = q[:, 1]
    y = q[:, 2]
    z = q[:, 3]

    R[:, 0, 0] = 1 - 2 * (y*y + z*z)
    R[:, 0, 1] = 2 * (x*y - w*z)
    R[:, 0, 2] = 2 * (x*z + w*y)
    R[:, 1, 0] = 2 * (x*y + w*z)
    R[:, 1, 1] = 1 - 2 * (x*x + z*z)
    R[:, 1, 2] = 2 * (y*z - w*x)
    R[:, 2, 0] = 2 * (x*z - w*y)
    R[:, 2, 1] = 2 * (y*z + w*x)
    R[:, 2, 2] = 1 - 2 * (x*x + y*y)
    return R


def get_projmat(znear:float, zfar:float, fovx:float, fovy:float) -> np.ndarray:
    """
    https://github.com/nerfstudio-project/nerfstudio/blob/9e33b437dff6df5a9579c04b1eba46640df88a96/nerfstudio/models/gaussian_splatting.py#L73
    """
    t = znear * math.tan(0.5 * fovy)
    b = -t
    r = znear * math.tan(0.5 * fovx)
    l = -r
    n = znear
    f = zfar

    zsign = 1.0
    return np.array(
        [[2 * n / (r - l), 0.0, (r + l) / (r - l), 0.0],
        [0.0, 2 * n / (t - b), (t + b) / (t - b), 0.0],
        [0.0, 0.0, (f + n) / (f - n), -1.0 * zsign * f * n / (f - n)],
        [0.0, 0.0, zsign, 0.0]],
    )

def get_viewmat(R:np.ndarray, t:np.ndarray) -> np.ndarray:
        """
        Construct world-to-view matrix using rotation matrix R and position t

        R is a shape `3,3` numpy array and `t` is a shape `3,` numpy array

        Returned viewmat is shaped `4,4`
        """
        V = np.eye(4)

        V[:3,:3] = R
        V[:3,3] = t

        return V
