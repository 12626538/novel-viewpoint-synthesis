import gsplat
import math
import numpy as np

import torch
from torch import nn,optim
import torch.nn.functional as F
import plyfile

from ..utils.camera import Camera
from ..utils import colmap_utils
from ..utils import batch_qvec2rotmat

class Gaussians(nn.Module):
    BLOCK_X, BLOCK_Y = 16, 16


    @classmethod
    def from_colmap(
        cls,
        path_to_points3D_file:str,
        device='cuda:0',
        **kwargs
    ) -> 'Gaussians':
        """
        Reads a `points3D.txt` file exported by COLMAP and creates a Gaussians instance with it
        by specifying the colors and means.

        Any additional kwargs will be passed to Gaussians.__init__
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

        Any additional kwargs will be passed to Gaussians.__init__
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
        scene_extend:float=1.,
        color_dim:int=3,
        act_scales=torch.exp,
        act_quats=F.normalize,
        act_colors=F.sigmoid,
        act_opacities=F.sigmoid,
        lr_position:float=0.00016,
        lr_scales:float=0.005,
        lr_quats:float=0.001,
        lr_colors:float=0.0025,
        lr_opacities:float=0.05,
        device='cuda:0',
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

        - `colors:Optional[torch.Tensor]` - Use this pre-defined parameter to initialize Gaussians. Shape `N,color_dim`.
        If not specified, initialize RGB uniformly in `0,1` (note that this will be passed through a sigmoid before rendering).

        - `opacities:Optional[torch.Tensor]` - Use this pre-defined parameter to initialize Gaussians. Shape `N,1`.
        If not specified, initialize to 1 (note that this will be passed through a sigmoid before rendering).

        - `scene_size:float` - Scale of the scene, used to initialize means if unspecified.

        - `color_dim:int` - Dimensionality of colors, default 3 (RGB)

        - `act_scales`,`act_quats`,`act_colors`,`act_opacities` - Activation functions, applied before rendering

        - `lr_position`,`lr_scales`,`lr_quats`,`lr_colors`,`lr_opacities` - Activation functions, applied before rendering

        where `N` is the number of points (either `num_points` or `means.shape[0]`)
        """
        # Init `torch.nn.Module` instance
        super().__init__()

        self.device = device

        # Use pre-defined means
        if means is not None:
            num_points = means.shape[0]

            # Update scene size to be at least a bounding box for the splats
            scene_extend = max(scene_extend, means.abs().max())

        # Use pre-defined number of points
        elif num_points is not None:
            means = scene_extend * 2*( torch.rand(num_points, 3, device=self.device) - .5 )

        # Raise error when neither means nor num_points is set
        else:
            raise ValueError("Either set `means` or `num_points` when initializing `Gaussians` instance")

        # Initialize scales
        if scales is None:
            scales = torch.log( torch.rand(num_points, 3, device=self.device) / (scene_extend*2) )

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
            colors = torch.rand(num_points, color_dim, device=self.device)

        # Initialize opacities
        if opacities is None:
            opacities = torch.ones((num_points, 1), device=self.device)

        # Sanity check: make sure all parameters have same number of points
        if not all(param.shape[0] == num_points for param in (means,scales,quats,colors,opacities)):
            raise ValueError("Not all parameters of Gaussians have the same first dimension")

        # Save as Parameters
        self.means = nn.Parameter(means.float())
        self.scales = nn.Parameter(scales.float())
        self.quats = nn.Parameter(quats.float())
        self.colors = nn.Parameter(colors.float())
        self.opacities = nn.Parameter(opacities.float())

        # Set up optimizer
        self.optimizer = optim.Adam(
            [
                {'params': [self.means], 'lr':lr_position, 'name': 'means'},
                {'params': [self.scales], 'lr':lr_scales, 'name': 'scales'},
                {'params': [self.quats], 'lr':lr_quats, 'name': 'quats'},
                {'params': [self.colors], 'lr':lr_colors, 'name': 'colors'},
                {'params': [self.opacities], 'lr':lr_opacities, 'name': 'opacities'},
            ]
        )

        # Set activation functions
        self.act_scales = act_scales
        self.act_quats = act_quats
        self.act_colors = act_colors
        self.act_opacities = act_opacities

        # Reset gradient stats, make sure no NameError happens
        self.reset_densification_stats()


    @property
    def num_points(self) -> int:
        return self.means.shape[0]


    def render(
            self,
            camera:Camera,
            glob_scale:float=1.0,
            bg:torch.Tensor=None,
        ) -> dict[str]:
        """
        Render a Camera instance using gaussian splatting

        For info on the gsplat library, see
        https://docs.gsplat.studio/

        Parameters
        - `camera:Camera` - Camera to render
        - `glob_scale:float=1.0` - Global scaling factor for the gaussians
        - `bg:Optional[torch.Tensor]=None` - Background to use, random if None

        Returns a **dict** with keys
        - `rendering:torch.Tensor` - Generated image, shape `H,W,C` where `C` is
        equal to `Gaussians.colors.shape[1]`
        - `xys:torch.Tensor` - 2D location of splats, shape `N,2` where `N` is
        equal to `Gaussians.num_points`.
        - `visibility_mask:torch.Tensor` - Mask of visible splats, shape `N`
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
            scales=self.act_scales( self.scales ),
            glob_scale=glob_scale,
            quats=self.act_quats( self.quats ),
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

        # Attempt to keep position gradients to update densification stats
        try:
            xys.retain_grad()
        except:
            pass

        # Generate image
        out_img = gsplat.rasterize_gaussians(
            xys=xys,
            depths=depths,
            radii=radii,
            conics=conics,
            num_tiles_hit=num_tiles_hit,
            colors=self.act_colors( self.colors ),
            opacity=self.act_opacities( self.opacities ),
            img_height=camera.H,
            img_width=camera.W,
            background=bg,
        )

        return {
            'rendering': out_img,
            'xys': xys,
            'radii': radii,
            'visibility_mask': (radii > 0).squeeze()
        }
    forward=render


    def reset_densification_stats(self) -> None:
        """
        Reset densification stats, automatically called after densifying
        """
        # Reset running average
        self._xys_grad_accum = torch.zeros((self.num_points,2), device=self.device)
        self._xys_grad_norm = torch.zeros((self.num_points,), device=self.device)

        # Reset maximum radii
        self._max_radii = torch.zeros(self.num_points, device=self.device)


    def update_densification_stats(
            self,
            xys:torch.FloatTensor,
            radii:torch.FloatTensor,
            visibility_mask:torch.BoolTensor,
        ) -> None:
        """
        Update internal running average of 2D positional gradient and maximum radii

        Must be called *after* the backwards pass of the optimizer (at least for the grads)
        """

        # Update running average of visible Gaussians
        if xys.grad is not None:
            self._xys_grad_accum[visibility_mask] += xys.grad[visibility_mask]
            self._xys_grad_norm[visibility_mask] += 1

        # Update max radii
        if radii is not None:
            self._max_radii[visibility_mask] = torch.max(
                self._max_radii[visibility_mask],
                radii[visibility_mask],
            )


    def densify(
            self,
            grad_threshold:float,
            max_scale:float,
            min_opacity:float,
            max_radius:float=None,
            N:int=2
        ):
        """
        Clone and Split Gaussians based on stats acumulated with rendering. Select
        Gaussians with a high viewspace gradient for cloning and splitting

        Clone: Increase volume and number of Gaussians in underconstructed areas
        by duplicating splats.

        Split: Increase number of Gaussians, but keep the total volume the same
        by replacing one splat by two smaller ones, based on the scale of the original.

        For both methods, for every selected splat, `N` new means are sampled using
        the original splat as a PDF. Opacity, color and rotation are duplicated for these
        new splats, and the scale is reduced by a factor `1/(.8*N)` for splats that are
        selected to be split (cloned splats will have the same scale as the original).

        Aditionally, discard any nearly transparent splat, as determined by `min_opacity`.
        Also discard any splat that was rendered with a radius larger than `max_radius` (if this
        arg is not None)

        Parameters:
        - `grad_threshold:float` - If the norm of the 2D spatial gradient of a splat exceeds
        this threshold, select it for splitting / cloning.
        - `max_scale:float` - If the scale of a splat exceeds this threshold, select it for
        splitting.
        - `min_opacity:float` - Remove any splat with a opacity lower than this value.
        - `max_radius:Optional[float]=None` - Remove any splat that was rendered with a radius larger than this value.
        If None, skip this step.
        - `N:int=2` - Number of splats to replace a selected splat with.
        """
        print("#"*20,f"\nDensifying... [started with {self.num_points} splats]")

        # Compute actual average
        grads = ( torch.linalg.norm(self._xys_grad_accum, dim=1) / self._xys_grad_norm ).to(self.device)
        grads[grads.isnan()] = 0.

        # Compute mask with gradient condition
        grad_cond = grads >= grad_threshold

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

        # Then the new mean is the original mean + (rotated) noise
        means = torch.bmm(rotmats, noise.unsqueeze(2)).squeeze() + self.means[grad_cond].repeat( N, 1 )

        # Also copy over the other splat parameters
        colors = self.colors[grad_cond].repeat( N, 1 )
        opacities = self.opacities[grad_cond].repeat( N, 1 )

        # For the split splats, also reduce the scale by 1.6 (as per the original implementation)
        split_cond = ( torch.max(scales,axis=1).values > max_scale ).squeeze()
        scales[split_cond] = scales[split_cond] / (0.8*N)

        print(f"{split_cond.sum()} splats selected for splitting")

        # Finally, from the unselected points, take those with a sufficient opacity
        prune_cond = grad_cond & (self.act_opacities( self.opacities ) <= min_opacity).squeeze()

        # If set, discard any splat that was rendered with a view radius larger than this
        if max_radius is not None:
            prune_cond = prune_cond & (self._max_radii > max_radius).to(self.device)

        print(f"Pruning {prune_cond.sum()} splats")

        # Add the unselected and split/pruned points together
        self.update_optimizer(
            dict_to_cat={
                'means':means,
                'scales':scales,
                'quats':quats,
                'colors':colors,
                'opacities':opacities,
            },
            prune_mask=prune_cond
        )

        print(f"Now has {self.num_points} splats")

        self.reset_densification_stats()

    def update_optimizer(
            self,
            dict_to_cat:dict[str,torch.Tensor],
            prune_mask:torch.BoolTensor,
        ) -> None:
        """
        Concatenate and prune existing parameters in optimizer

        From https://github.com/graphdeco-inria/gaussian-splatting/blob/2eee0e26d2d5fd00ec462df47752223952f6bf4e/scene/gaussian_model.py#L307
        """


        keep_mask = ~prune_mask

        # Iterate parameter groups
        for group in self.optimizer.param_groups:

            # Get tensor to concatenate, or empty if nothing to add
            extension_tensor = dict_to_cat.get(group["name"],torch.empty(0))

            # Update stored state
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:

                # Initialize extension stats as zeros
                stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"][keep_mask], torch.zeros_like(extension_tensor)), dim=0)
                stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"][keep_mask], torch.zeros_like(extension_tensor)), dim=0)

                # Update parameter/stored state
                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0][keep_mask], extension_tensor), dim=0))
                self.optimizer.state[group['params'][0]] = stored_state

            # Update parameter
            else:
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0][keep_mask], extension_tensor), dim=0))

            # Update Gaussians.means etc to updated group parameter
            setattr(self, group['name'], group['params'][0])


    def reset_opacity(self):
        """
        Clip all opacities at 0.01

        Because opacities get fed through a sigmoid, set the value to `-4.59512`
        s.t. `sigmoid(-4.59512) = 0.01`
        """
        self.opacities.clamp_max_(-4.59512)
