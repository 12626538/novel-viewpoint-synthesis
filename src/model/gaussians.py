import gsplat
import math

import torch
from torch import nn

from ..utils.camera import Camera
from ..utils.colmap_utils import read_points3D

class Gaussians(nn.Module):
    BLOCK_X, BLOCK_Y = 16, 16

    @classmethod
    def from_colmap(
        cls,
        path_to_points3D_file:str,
        device='cuda:0',
        *args,**kwargs
    ) -> 'Gaussians':

        means, colors = read_points3D(path_to_points3D_file)
        means = torch.from_numpy(means).to(device=device)
        colors = torch.from_numpy(colors).to(device=device)

        num_points = means.shape[0]

        return Gaussians(
            num_points=num_points,
            device=device,
            means=means,
            colors=colors,
            *args,**kwargs
        )

    def __init__(
        self,
        num_points=100_000,
        scene_size=10,
        device='cuda:0',
        means=None,
        scales=None,
        quats=None,
        colors=None,
        opacities=None,
    ):
        super().__init__()
        self.num_points = num_points
        self.device = device

        if means is None:
            means = scene_size * 2 *torch.rand(self.num_points, 3, device=self.device) - scene_size

        if scales is None:
            scales = torch.rand(self.num_points, 3, device=self.device) / (scene_size*2.)

        if quats is None:
            u = torch.rand(self.num_points, 1, device=self.device)
            v = torch.rand(self.num_points, 1, device=self.device)
            w = torch.rand(self.num_points, 1, device=self.device)
            quats = torch.cat(
                [
                    torch.sqrt(1.0 - u) * torch.sin(2.0 * math.pi * v),
                    torch.sqrt(1.0 - u) * torch.cos(2.0 * math.pi * v),
                    torch.sqrt(u) * torch.sin(2.0 * math.pi * w),
                    torch.sqrt(u) * torch.cos(2.0 * math.pi * w),
                ],
                -1,
            )

        if colors is None:
            colors = torch.rand(self.num_points, 3, device=self.device)

        if opacities is None:
            opacities = torch.ones((self.num_points, 1), device=self.device)/2.

        self.means = nn.Parameter(means.float())
        self.scales = nn.Parameter(scales.float())
        self.quats = nn.Parameter(quats.float())
        self.colors = nn.Parameter(colors.float())
        self.opacities = nn.Parameter(opacities.float())

    def render(
            self,
            camera:Camera,
            glob_scale:float=1.0,
            bg:torch.Tensor=None,
        ):
        """
        Render a Camera instance using gaussian splatting

        For info on the gsplat library, see
        https://docs.gsplat.studio/

        Parameters
        - `camera:Camera` - Camera to render
        - `glob_scale:float=1.0` - Global scaling factor for the gaussians
        - `bg:Optional[torch.Tensor]=None` - Background to use, random if None

        Returns
        - `out_img:torch.Tensor` - Generated image, shape `H,W,C` where `C` is
        equal to `Gaussians.rgbs.shape[1]`
        """

        # Set up number of tiles in X,Y,Z direction
        tile_bounds = (
            (camera.W + self.BLOCK_X - 1) // self.BLOCK_X,
            (camera.H + self.BLOCK_Y - 1) // self.BLOCK_Y,
            1,
        )

        # If unspecified generate random background
        if bg is None:
            bg = torch.rand(3, device=self.device)

        # Project Gaussians from 3D to 2D
        xys, depths, radii, conics, num_tiles_hit, cov3d = gsplat.project_gaussians(
            means3d=self.means,
            scales=self.scales,
            glob_scale=glob_scale,
            quats=self.quats,
            viewmat=camera.viewmat,
            projmat=camera.projmat @ camera.viewmat,
            fx=camera.fx,
            fy=camera.fy,
            cx=camera.W / 2,
            cy=camera.H / 2,
            img_width=camera.W,
            img_height=camera.H,
            tile_bounds=tile_bounds
        )

        # Generate image
        out_img = gsplat.rasterize_gaussians(
            xys=xys,
            depths=depths,
            radii=radii,
            conics=conics,
            num_tiles_hit=num_tiles_hit,
            colors=torch.sigmoid(self.colors),
            opacity=torch.sigmoid(self.opacities),
            img_height=camera.H,
            img_width=camera.W,
            background=bg,
        )

        return out_img
    forward=render
