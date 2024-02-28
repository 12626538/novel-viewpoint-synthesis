import torch
import math
import numpy as np

from src.utils import fov2focal,get_projmat,get_viewmat

class Camera(object):
    """
    Base class for cameras

    Comporable to the JaxCam class used for (for example) ZipNeRF.
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
            H:int=1920,W:int=1080,
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

        # Intrinsics
        self.znear = znear
        self.zfar = zfar

        self.fovx = fovx
        self.fovy = fovy

        self.cx_frac = cx_frac
        self.cy_frac = cy_frac

        self.zsign = zsign

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
            fovx=self.fovx,fovy=self.fovy,
            zsign=self.zsign,
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
    def cx(self) -> float:
        """Principle point in X direction"""
        return self.cx_frac * self.W


    @property
    def cy(self) -> float:
        """Principle point in Y direction"""
        return self.cy_frac * self.H


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
            "Camera instance\n"
            f"\tname: '{self.name}'\n"
            f"\tDevice: {self.device}\n"
            f"\tH,W: {self.H,self.W}\n"
            f"\tcx,cy: {self.cx:.2f}, {self.cy:.2f}\n"
            f"\tfx,fy: {self.fx:.2f}, {self.fy:2f}\n"
            f"\tFoV X,Y (in degrees): {np.rad2deg(self.fovx):.2f}, {np.rad2deg(self.fovy):.2f}\n"
            f"\tHas gt image? {self.gt_image is not None}\n"
            f"\tviewmat:\n{self.viewmat}\n"
            f"\tprojmat:\n{self.projmat}\n"
        )

    def project_points(self, points:torch.FloatTensor, res_agnostic:bool=True, visible_only:bool=True) -> torch.FloatTensor:
        """
        Project a collection of 3D, world-coordinate points to 2D pixel
        locations.

        If `res_agnostic` is set to True, returned points will be in the
        interval `[0,1]x[0,1]` to be resolution agnostic. This scaling is done
        perserving the aspect ratio by scaling by the max resolution in X or Y
        direction.

        This method is differientable w.r.t. `points`.

        Returned tensor will be on the same device as input device

        Parameters:
        - `points:torch.Tensor` - 3D points, shape `N,3`
        - `res_agnostic:bool` - Rescale pixel coordinates to be in `[0,1]` to be
            resolution agnostic.
        - `visible_only:bool` - Only return points inside view frustum

        Returns:
        - `pixels:torch.FloatTensor` - 2D points, shape `N,2`, dtype `float`.
        """

        # Export to this device
        in_device = points.device

        # Convert euclidean to homogeneous
        hom_points = torch.hstack((
            points.to(device=self.device),
            torch.ones(points.shape[0], 1, device=self.device)
        ))

        # Convert world to camera coordinates
        cam_points = hom_points @ (self.viewmat).T

        # Project points
        view_points = cam_points @ (self.projmat).T

        # Scale by hom coeff
        view_points /= view_points[...,-1, None]

        if visible_only:
            # Prune points outside view frustum
            in_view = (
                ( view_points[..., 0] >= 0 )
                & (view_points[..., 0] < self.W )
                & ( view_points[..., 1] >= 0 )
                & (view_points[..., 1] < self.H )
                & (view_points[..., 2] >= self.znear )
                & (view_points[..., 2] < self.zfar )
            )

            view_points = view_points[in_view]

        # Scale by max resolution to be resolution agnostic
        if res_agnostic:
            view_points /= max(self.H,self.W)

        return view_points.to(device=in_device)
