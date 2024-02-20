import torch
import numpy as np
import math
from PIL import Image
from torchvision import transforms

def sigmoid_inv(x):
    return np.log(x/(1-x))

def qvec2rotmat_np(qvec:np.ndarray) -> np.ndarray:
    """
    Convert quaternion `qvec=[w,x,y,z]` to 3x3 rotation matrix
    """
    assert qvec.shape == (4,), qvec.shape
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
    """
    From https://github.com/graphdeco-inria/gaussian-splatting/blob/main/utils/graphics_utils.py
    """
    return 2*math.atan(pixels/(2*focal))

def fov2focal(fov:float, pixels:int) -> float:
    """
    From https://github.com/graphdeco-inria/gaussian-splatting/blob/main/utils/graphics_utils.py
    """
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

def qvec2rotmat(q:torch.Tensor) -> torch.Tensor:
    """
    Convert quaternion vector(s) to rotation matrices

    Works both batches as unbatched

    Diffirientiable

    Code from
    https://github.com/graphdeco-inria/gaussian-splatting/blob/2eee0e26d2d5fd00ec462df47752223952f6bf4e/utils/general_utils.py#L78

    Parameters:
    - `batch_qvec:torch.Tensor` of shape `...,4` in WXYZ format

    Returns
    - `batch_rotmat:torch.Tensor` of shape `...,3,3`
    """

    assert q.shape[-1]==4, q.shape

    # norm = torch.sqrt(batch_qvec[:,0]*r[:,0] + r[:,1]*r[:,1] + r[:,2]*r[:,2] + r[:,3]*r[:,3])
    q = q / torch.linalg.norm(q, dim=-1,keepdim=True)

    R = torch.zeros( q.shape[:-1]+(3, 3), device=q.device )

    w,x,y,z = torch.unbind(q,dim=-1)

    R[..., 0, 0] = 1 - 2 * (y*y + z*z)
    R[..., 0, 1] = 2 * (x*y - w*z)
    R[..., 0, 2] = 2 * (x*z + w*y)
    R[..., 1, 0] = 2 * (x*y + w*z)
    R[..., 1, 1] = 1 - 2 * (x*x + z*z)
    R[..., 1, 2] = 2 * (y*z - w*x)
    R[..., 2, 0] = 2 * (x*z - w*y)
    R[..., 2, 1] = 2 * (y*z + w*x)
    R[..., 2, 2] = 1 - 2 * (x*x + y*y)
    return R


def get_projmat(znear:float, zfar:float, fovx:float, fovy:float) -> np.ndarray:
    """
    From http://www.songho.ca/opengl/gl_projectionmatrix.html
    """
    n = znear
    f = zfar

    a = 1 / math.tan( fovx / 2 )
    b = 1 / math.tan( fovy / 2 )

    zsign = 1.0
    return np.array([
        [  a, 0.0,          0.0,                 0.0],
        [0.0,   b,          0.0,                 0.0],
        [0.0, 0.0, -(f+n)/(f-n), zsign*(2*f*n)/(f-n)],
        [0.0, 0.0,        zsign,                 0.0]
    ])

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
