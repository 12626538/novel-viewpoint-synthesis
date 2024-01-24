import gsplat
import math
import numpy as np

import torch
from torch import nn
import plyfile

from ..utils.camera import Camera
from ..utils import colmap_utils
from ..utils import batch_qvec2rotmat

class Gaussians(nn.Module):
    BLOCK_X, BLOCK_Y = 16, 16

    act_scales = torch.exp
    act_colors = torch.sigmoid
    act_opacities = torch.sigmoid


    @classmethod
    def from_colmap(
        cls,
        path_to_points3D_file:str,
        device='cuda:0',
        **kwargs
    ) -> 'Gaussians':
        """
        Reads a `points3D.txt` file exported by COLMAP and creates a Gaussians instance with it
        by specifying the colors and means. Opacities and rotations will still be random
        """

        means, colors = colmap_utils.read_points3D(path_to_points3D_file)
        means = torch.from_numpy(means).to(device=device)
        colors = torch.from_numpy(colors).to(device=device)

        return Gaussians(
            means=means,
            colors=colors,
            device=device,
            **kwargs
        )


    @classmethod
    def from_ply(
        cls,
        path_to_ply_file:str,
        device='cuda:0',
        **kwargs
    ) -> 'Gaussians':
        """
        Read `.ply` file and initialize Gaussians using positions and colors
        from PlyData.
        """
        # Read ply data
        plydata = plyfile.PlyData.read(path_to_ply_file)

        # Extract positions as np array
        vertices = plydata['vertex']
        positions = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T

        # Extract colors as RGB
        colors = np.vstack([vertices['red'], vertices['green'], vertices['blue']]).T / 255.0

        # Create Gaussians instance
        return Gaussians(
            means=torch.from_numpy(positions).to(device),
            colors=torch.from_numpy(colors).to(device),
            device=device,
            **kwargs
        )


    def __init__(
        self,
        num_points:int=None,
        means:torch.Tensor=None,
        scales:torch.Tensor=None,
        quats:torch.Tensor=None,
        colors:torch.Tensor=None,
        opacities:torch.Tensor=None,
        scene_size:float=1.,
        max_scale:float=0.01,
        device='cuda:0',
        grad_threshold:float=0.0002,
        min_opacity:float=0.005,
    ):
        """
        Set up Gaussians instance

        Will raise ValueError if neither of `num_points`,`means` are set

        Will raise ValueError if for some reason parameters do not share the same first dimension size

        Parameters:
        - `num_pints:Optional[int]` - Number of points to initialize, if not set,
        use `means.shape[0]` to derive the number of points.

        - `means:Optional[torch.Tensor]` - Use this pre-defined parameter to initialize Gaussians. Shape `N,3`.
        If not specified, use `scene_size` and `num_points` to uniformly initialize means in range
        `-scene_size, scene_size` in XYZ directions.

        - `scales:Optional[torch.Tensor]` - Use this pre-defined parameter to initialize Gaussians. Shape `N,3`.
        If not specified, initialize uniformly IID in interval `0, -ln(2*scene_size)` (note that this will be passed
        through an exp activation function)

        - `quats:Optional[torch.Tensor]` - Use this pre-defined parameter to initialize Gaussians. Shape `N,4`.
        If not specified, initialize uniformly random rotation.

        - `colors:Optional[torch.Tensor]` - Use this pre-defined parameter to initialize Gaussians. Shape `N,C`.
        If not specified, initialize RGB uniformly in `0,1` (note that this will be passed through a sigmoid before rendering).

        - `opacities:Optional[torch.Tensor]` - Use this pre-defined parameter to initialize Gaussians. Shape `N,1`.
        If not specified, initialize to 1 (note that this will be passed through a sigmoid before rendering).

        - `scene_size:float` - Scale of the scene, used to initialize means if unspecified.

        - `grad_threshold:float=.0002` - Gradient threshold for densification

        - `max_scale:float=0.01` - When densifying, discard any splat with a scale larger than this

        - `min_opacity:float=0.005` - When densifying, prune splats with a opacity lower than this

        where `N` is the number of points (either `num_points` or `means.shape[0]`)
        and `C` is the feature dimension of colors, default is `3` (RGB).
        """
        # Init `torch.nn.Module` instance
        super().__init__()

        self.device = device

        self.grad_threshold = grad_threshold
        self.min_opacity = min_opacity

        # Use pre-defined means
        if means is not None:
            num_points = means.shape[0]

            # Update scene size to be at least a bounding box for the splats
            scene_size = max(scene_size, means.abs().max())

        # Use pre-defined number of points
        elif num_points is not None:
            means = scene_size * 2*( torch.rand(num_points, 3, device=self.device) - .5 )

        # Raise error when neither means nor num_points is set
        else:
            raise ValueError("Either set `means` or `num_points` when initializing `Gaussians` instance")

        # Set max density
        self.max_scale = max_scale

        # Initialize scales
        if scales is None:
            scales = torch.log( torch.rand(num_points, 3, device=self.device) / (scene_size*2) )

        # Initialize rotation
        if quats is None:
            u = torch.rand(num_points, 1, device=self.device)
            v = torch.rand(num_points, 1, device=self.device)
            w = torch.rand(num_points, 1, device=self.device)
            quats = torch.cat(
                [
                    torch.sqrt(1.0 - u) * torch.sin(2.0 * math.pi * v),
                    torch.sqrt(1.0 - u) * torch.cos(2.0 * math.pi * v),
                    torch.sqrt(u) * torch.sin(2.0 * math.pi * w),
                    torch.sqrt(u) * torch.cos(2.0 * math.pi * w),
                ],
                -1,
            )

        # Initialize colors
        if colors is None:
            colors = torch.rand(num_points, 3, device=self.device)

        # Initialize opacities
        if opacities is None:
            opacities = torch.ones((num_points, 1), device=self.device)

        # Sanity check: make sure all parameters have same number of points
        if not all(param.shape[0] == num_points for param in (means,scales,quats,colors,opacities)):
            raise ValueError("Not all parameters of Gaussians have the same first dimension")


        # Save parameters to model
        self.set_parameters(
            means=means,
            scales=scales,
            quats=quats,
            colors=colors,
            opacities=opacities
        )

        # Reset gradient stats, make sure no NameError happens
        self.reset_densification_stats()


    def set_parameters(self,means,scales,quats,colors,opacities):
        """
        Set parameters as Parameters to model

        This method is shared by the __init__ and densify fuction
        """
        # Save parameters as Parameters
        self.means = nn.Parameter(means.float())
        self.scales = nn.Parameter(scales.float())
        self.quats = nn.Parameter(quats.float())
        self.colors = nn.Parameter(colors.float())
        self.opacities = nn.Parameter(opacities.float())


    @property
    def num_points(self) -> int:
        return self.means.shape[0]


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
        self._last_xys, depths, self._last_radii, conics, num_tiles_hit, cov3d = gsplat.project_gaussians(
            means3d=self.means,
            scales=self.act_scales( self.scales ),
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

        # Save a mask of visible Gaussians, used for densification
        self._last_visible = ( self._last_radii.detach().cpu() > 0 ).squeeze()

        # Attempt to keep position gradients
        try:
            self._last_xys.retain_grad()
        except:
            pass

        # Generate image
        out_img = gsplat.rasterize_gaussians(
            xys=self._last_xys,
            depths=depths,
            radii=self._last_radii,
            conics=conics,
            num_tiles_hit=num_tiles_hit,
            colors=self.act_colors(self.colors),
            opacity=self.act_opacities(self.opacities),
            img_height=camera.H,
            img_width=camera.W,
            background=bg,
        )

        return out_img
    forward=render


    def reset_densification_stats(self) -> None:
        """
        Reset densification stats, automatically called after densifying
        """
        # Disable `update_grad_stats` until a `render` call has been made
        self._last_xys = None
        self._last_visible = torch.full((self.num_points,), False)

        # Reset running average
        self._xys_grad = torch.zeros((self.num_points,2))
        self._xys_grad_norm = torch.zeros((self.num_points,))

        # Reset maximum radii
        self._last_radii = None
        self._max_radii = torch.zeros(self.num_points)


    def update_densification_stats(self) -> None:
        """
        Update internal running average of 2D positional gradient and maximum radii

        Must be called *after* the backwards pass of the optimizer (at least for the grads)
        """

        # Update running average of visible Gaussians
        if self._last_xys is not None and self._last_xys.grad is not None:
            self._xys_grad[self._last_visible] += self._last_xys.grad.detach().cpu()[self._last_visible]
            self._xys_grad_norm[self._last_visible] += 1

        # Update max radii
        if self._last_radii is not None:
            self._max_radii[self._last_visible] = torch.max(
                self._max_radii[self._last_visible],
                self._last_radii.detach().cpu()[self._last_visible],
            )


    def densify(self,N=2):
        """
        Clone and Split Gaussians based on stats acumulated with rendering. Select
        Gaussians with a high viewspace gradient for cloning and splitting

        Clone: Increase volume and number of Gaussians in underconstructed areas
        by duplicating splats.

        Split: Increase number of Gaussians, but keep the total volume the same
        by replacing one splat by two smaller ones, based on the scale of the original.

        Parameters:
        - `N:int=2` - Number of splats to replace a selected splat with.
        """
        print("#"*20,f"\nDensifying... [started with {self.num_points} splats]")
        # Compute actual average
        grads = ( torch.linalg.norm(self._xys_grad, dim=1) / self._xys_grad_norm ).to(self.device)
        grads[grads.isnan()] = 0.

        # Compute mask with gradient condition
        grad_cond = grads >= self.grad_threshold

        print(f"{grad_cond.sum()} splats selected for cloning/splitting")

        # Create `N` copies of each selected gplat by taking some noise based on the scale
        scales = self.scales[grad_cond].repeat( N, 1)
        noise = torch.normal(
            mean=torch.zeros((scales.shape[0],3), device=self.device),
            std=self.act_scales(scales)
        )

        # Rotating that noise to also be oriented like the original splat
        quats = self.quats[grad_cond].repeat( N, 1 )
        rotmats = batch_qvec2rotmat(quats)

        # Then the new mean is the original mean + rotated noise
        means = torch.bmm(rotmats, noise.unsqueeze(2)).squeeze() + self.means[grad_cond].repeat( N, 1 )

        # Also copy over the other splat parameters
        colors = self.colors[grad_cond].repeat( N, 1 )
        opacities = self.opacities[grad_cond].repeat( N, 1 )

        # For the split splats, also reduce the scale by 1.6 (as per the original implementation)
        split_cond = ( torch.max(scales,axis=1).values > self.max_scale ).squeeze()
        scales[split_cond] = scales[split_cond] / (0.8*N)

        print(f"{split_cond.sum()} splats selected for splitting")

        # Finally, from the unselected points, take those with a sufficient opacity
        keep_cond = ~grad_cond & (self.opacities > self.min_opacity).squeeze()

        print(f"Not touching {keep_cond.sum()} splats")

        # Add the unselected and split/pruned points together
        self.set_parameters(
            means=torch.concat( ( self.means[keep_cond], means ) ),
            scales=torch.concat( ( self.scales[keep_cond], scales ) ),
            quats=torch.concat( ( self.quats[keep_cond], quats ) ),
            colors=torch.concat( ( self.colors[keep_cond], colors ) ),
            opacities=torch.concat( ( self.opacities[keep_cond], opacities ) )
        )

        print(f"Now has {self.num_points} splats")

        self.reset_densification_stats()
