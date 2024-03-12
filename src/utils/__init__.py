import torch
import numpy as np
import math
from PIL import Image
from torchvision import transforms

def knn_scale(x:np.ndarray, k:int=3):
    """
    Use K-Nearest Neighbours to estimate the scale of a point.

    The scale of a point in `x` is taken as the average distance to the nearest
    `k` neighbours of that point (excluding itself).

    Requires sklearn package.

    Parameters:
    - `x:np.ndarray` of shape `N,D` where `N` is number of points and `D` is
        dimensionality of the data.
    - `k:int=3` number of neighbours to consider

    Returnes:
    - `scales:np.ndarray` of shape `N,D` where each row has the same value
    """
    from sklearn.neighbors import NearestNeighbors

    D = x.shape[-1]
    nn_model = NearestNeighbors(n_neighbors=k + 1, algorithm="auto", metric="euclidean").fit(x)

    # Find the k-nearest neighbors
    distances, _ = nn_model.kneighbors(x)

    # Exclude the point itself from the result and take the average distance
    distances = distances[:, 1:].mean(dim=-1, keepdims=True)

    return np.tile(distances, (1,D))

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


def get_projmat(znear:float, zfar:float, fovx:float, fovy:float, zsign:float) -> np.ndarray:
    """
    From http://www.songho.ca/opengl/gl_projectionmatrix.html

    Using the simplification on the first two diagonal entries:
    2 fx / w = 2 w / (2 w tan( fovx / 2 ) ) = 1 / tan( fovx / 2 )
    2 fy / h = 2 h / (2 h tan( fovy / 2 ) ) = 1 / tan( fovy / 2 )
    """
    t = znear * math.tan(0.5 * fovy)
    b = -t
    r = znear * math.tan(0.5 * fovx)
    l = -r
    n = znear
    f = zfar
    return np.array(
        [
            [2 * n / (r - l), 0.0, (r + l) / (r - l), 0.0],
            [0.0, 2 * n / (t - b), (t + b) / (t - b), 0.0],
            [0.0, 0.0, (f + n) / (f - n), -1.0 * f * n / (f - n)],
            [0.0, 0.0, 1.0, 0.0],
        ],
    )
    n = znear
    f = zfar

    a = 1 / math.tan( fovx / 2 )
    b = 1 / math.tan( fovy / 2 )

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
