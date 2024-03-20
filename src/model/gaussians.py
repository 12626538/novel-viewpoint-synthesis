from dataclasses import dataclass
import os
import gsplat
import math
import numpy as np

import torch
from torch import nn,optim
import torch.nn.functional as F
from plyfile import PlyElement, PlyData

from src.camera import Camera
from src.utils import colmap_utils, qvec2rotmat, sigmoid_inv, lr_utils, knn_scale

# TODO:
from src.model.blur import Blurrer

# An instance of this is returned by the `Gaussians.render` method
@dataclass
class RenderPackage:
    """
    A simple dataclass containing all relevant properties of the
    `Gaussians.render` method.
    """
    rendering:torch.Tensor # Shape C,H,W
    xys:torch.Tensor # Shape N,3
    radii:torch.Tensor # Shape N
    visibility_mask:torch.BoolTensor # Shape N
    alpha:torch.Tensor # Shape H,W

class Gaussians(nn.Module):
    """
    TODO: write a motivational introduction about 3DGS
    """

    @classmethod
    def from_colmap(
        cls,
        path_to_points3D_file:str,
        device='cuda',
        sh_degree:int=3,
        **kwargs
    ) -> 'Gaussians':
        """
        Reads a `points3D.txt` file exported by COLMAP and creates a Gaussians instance with it
        by specifying the colors and means.

        Any additional kwargs will be passed to Gaussians.__init__
        """

        means, colors = colmap_utils.read_points3D(path_to_points3D_file)
        means = torch.from_numpy(means).to(device=device)

        if sh_degree == 0:
            # Account for sigmoid activation function for the colors
            # Clip from [0,1] to (0,1) to prevent division by zero errors
            colors = sigmoid_inv(colors.clip(1e-7, 1-1e-7))
            c = colors.shape[-1]
            colors = torch.from_numpy(colors.reshape(-1,1,c)).to(device=device, dtype=torch.float32)
        else:
            n,c = colors.shape
            d = (sh_degree+1)**2

            # Account for SH band 0
            # C0 from https://github.com/nerfstudio-project/gsplat/blob/main/gsplat/cuda/csrc/sh.cuh
            colors = (colors - 0.5) / 0.28209479177387814

            _colors = np.zeros((n,d,c))
            _colors[:,0] = colors
            colors = torch.from_numpy(_colors).to(device=device, dtype=torch.float32)

        return Gaussians(
            means=means,
            colors=colors,
            device=device,
            sh_degree=sh_degree,
            **kwargs
        )


    @classmethod
    def from_ply(
        cls,
        path_to_ply_file:str,
        device='cuda',
        **kwargs
    ) -> 'Gaussians':
        """
        Read `.ply` file and initialize Gaussians using positions and colors
        from PlyData.

        Any additional kwargs will be passed to Gaussians.__init__

        Raises:
        - ValueError if no properties interpretable as means are found (see below)
        - ValueError if colors are badly formatted. This is either because not all
            colors in `color_1, ..., color_1_C, ..., color_D_C` are present, or
            `D` does not satisfy `D=(sh_degree+1)^2` for some `sh_degree`.

        Expects the following formats:
        - Means can either be with properties `x,y,z` or `pos_1,pos_2,pos_3`.
            One of these is required, raise ValueError of neither are present.
        - Scales should be formatted as `scale_1,scale_2,scale_3`
        - Rotations (quaternions) can either be formatted as
            `rot_1,rot_2,rot_3,rot_4` or `quat_w,quat_x,quat_y,quat_z`.
        - Colors are a `D x C` dimensional feature vector per splat, where
            `D=(sh_degree+1)^2` are spherical harmonic features and `C` is some
            number of colors. Accepted formats are:
            - `red,green,blue` for `sh_degree=0` and `C=3`.
            - `color_1, ..., color_C` for `sh_degree=0` some `C`
            - `color_1, ..., color_1_C, ..., color_D_C` for some `D` and `C`.
                Raises ValueError if not all colors are present or `D` does not
                satisfy `D=(sh_degree+1)^2` for some `sh_degree`.
        - Opacities can either be in a `opacities` or `opacity` field.
        """
        # Read ply data
        plydata = PlyData.read(path_to_ply_file)

        # Extract positions as np array
        vertex:PlyElement = plydata['vertex']

        # Get properties as a set of strings for easy subset checking
        properties = {prop.name for prop in vertex.properties}

        # convert either xyz or pos_1/2/3 to means
        if properties.issuperset('xyz'):
            means = np.stack([vertex['x'], vertex['y'], vertex['z']], axis=1)
        elif properties.issuperset({'pos_1','pos_2','pos_3'}):
            means = np.stack([vertex['pos_1'], vertex['pos_2'], vertex['pos_3']], axis=1)
        else:
            raise ValueError("Cannot read .ply data, vertex needs either xyz or pos_1/2/3 properties")
        kwargs['means'] = torch.from_numpy(means).to(device)

        # Get scales from properties
        if properties.issuperset({"scale_1","scale_2","scale_3"}):
            scales = np.stack([vertex['scale_1'], vertex['scale_2'], vertex['scale_3']], axis=1)
            kwargs['scales'] = torch.from_numpy(scales).to(device)

        # Otherwise, use K-NN to get the average distances to the nearest K neighbours and use that as scale
        else:
            # Get scales using KNN
            scales = knn_scale(means)
            # Take the log to account for exp activation function
            kwargs['scales'] = torch.from_numpy(scales).to(device).log()

        # Convert rot_1/2/3/4 to quaternions
        if properties.issuperset({"rot_1","rot_2","rot_3","rot_4"}):
            quats = np.stack([vertex['rot_1'], vertex['rot_2'], vertex['rot_3'], vertex['rot_4']], axis=1)
            kwargs['quats'] = torch.from_numpy(quats).to(device)
        # Convert rot_w/x/y/z to quaternions
        elif properties.issuperset({"quat_w","quat_x","quat_y","quat_z"}):
            quats = np.stack([vertex['quat_w'], vertex['rot_x'], vertex['rot_y'], vertex['rot_z']], axis=1)
            kwargs['quats'] = torch.from_numpy(quats).to(device)

        # Extract colors
        colors = None
        if properties.issuperset({'red','green','blue'}):
            colors = np.stack([vertex['red'], vertex['green'], vertex['blue']], axis=1, dtype=np.float32) / 255.

            kwargs['sh_degree'] = kwargs.get('sh_degree',2)
            D = (kwargs['sh_degree']+1)**2

            # If sh_degree > 0, account for 0-band SH color
            if D>1:
                # C0 from https://github.com/nerfstudio-project/gsplat/blob/main/gsplat/cuda/csrc/sh.cuh
                colors = (colors - 0.5) / 0.28209479177387814
            else:
                # Otherwise, account for sigmoid activation
                colors = sigmoid_inv( colors.clip(1e-7, 1-1e-7) )

            features = np.zeros((colors.shape[0],D, colors.shape[1]))
            features[:,0,:] = colors

            kwargs['colors'] = torch.from_numpy(features).to(device)

        # Convert color_1,color_2,...,color_C to NxC numpy array
        elif 'color_1' in properties:

            # BUG: sorted will output [color_1, color_10, color_2, ...]
            cols = sorted([prop for prop in properties if prop.startswith("color_")])
            colors = np.stack([vertex[c] for c in cols], axis=1)
            kwargs['colors'] = torch.from_numpy(colors).to(device).reshape(-1, 1, len(cols))
            kwargs['sh_degree']=0

        # Convert color_1_1, ..., color_1_C, ..., color_D_C to NxDxC numpy array
        elif 'color_1_1' in properties:

            # Find the D and C in color_D_C
            D = max(int(color.split('_')[1]) for color in properties if color.startswith('color_'))
            C = max(int(color.split('_')[2]) for color in properties if color.startswith('color_'))

            # Set up list of colors color_1_1, ..., color_1_C, ..., color_D_C
            cols = [f'color_{d+1}_{c+1}' for d in range(D) for c in range(C)]

            # Sanity check: make sure all colors and degrees are represented
            if not properties.issuperset(cols):
                raise ValueError("Unexpected configuration of colors. Expected 'color_d_c' "
                                 f"for all d=1,...,{D} and c=1,...,{C}, missing {set(cols)-properties}")

            # Sanity check, make sure D=(d+1)^2 for some sh degree d
            sh_degree:float = np.sqrt(D)-1
            if not sh_degree.is_integer():
                raise ValueError("Unexpected number of features per color, "
                                 f"got D={D}, expected D=(d+1)^2 for some degree d.")

            # Stack colors together
            colors = np.stack([vertex[c] for c in cols],axis=1).reshape(-1,D,C)

            kwargs['sh_degree'] = int(sh_degree)
            kwargs['colors'] = torch.from_numpy( colors ).to(device)

        # Extract opacities
        if 'opacities' in properties:
            kwargs['opacities'] = torch.from_numpy( vertex['opacities'].reshape(-1,1) ).to(device)
        elif 'opacity' in properties:
            kwargs['opacities'] = torch.from_numpy( vertex['opacity'].reshape(-1,1) ).to(device)

        # Create Gaussians instance
        return Gaussians(
            device=device,
            **kwargs,
        )


    def to_ply(self, path:str,overwrite:bool=False):
        """
        Save Gaussians instance in .ply format

        If `path` does not end with `.ply`, `path` is treated as a directory and data will be saved at
        `[path]/point_cloud.ply`.

        Raise PermissionError if `path` already exists. This is surpressed if `overwrite` is set `True`.

        ply format:
        - Means are `pos_1`,`pos_2`,`pos_3`
        - Scales are `scale_1`,`scale_2`,`scale_3`
        - Rotation as quaternions `rot_1`,`rot_2`,`rot_3`,`rot_4` in WXYZ format
        - Color as `color_d_c` for `d=1,...,D` and `c=1,...,C` where `D,C == colors.shape[-2:]`
        - Opacities as `opacity`

        Same format is used to read in the classmethod `Gaussians.from_ply`.
        """

        # If not a filename, treat as directory
        if not path.endswith(".ply"):
            path = os.path.join(path,"point_cloud.ply")

        # Overwriting protection
        if os.path.isfile(path) and not overwrite:
            raise PermissionError(f"Saving to {path} not possible, file already exists")

        # Create directory if it does not exists
        os.makedirs(os.path.dirname(path), exist_ok=True)

        # Fields to use
        fields = ['x','y','z'] \
            + [f'scale_{s+1}' for s in range(self.scales.shape[1])] \
            + [f'rot_{c+1}' for c in range(self.quats.shape[1])] \
            + [f'color_{d+1}_{c+1}' for d in range(self.colors.shape[1]) for c in range(self.colors.shape[2])] \
            + ['opacity',]

        # Set up output matrix
        dtypes = [(field,'f4') for field in fields]
        data = np.empty(self.means.shape[0], dtype=dtypes)

        # Create big matrix with all features
        features = np.concatenate(
            [tensor.flatten(start_dim=1).detach().cpu().numpy()
             for tensor in ( self.means, self.scales, self.quats, self.colors, self.opacities )],
            axis=1
        )

        # Put features as a list of tuples in output
        data[:] = list(map(tuple, features))

        # Export as ply element
        el = PlyElement.describe(data, 'vertex')
        PlyData([el]).write(path)


    def __init__(
        self,
        num_points:int=None,
        means:torch.Tensor=None,
        scales:torch.Tensor=None,
        quats:torch.Tensor=None,
        colors:torch.Tensor=None,
        opacities:torch.Tensor=None,
        scene_extend:float=1.,
        sh_degree:int=3,
        sh_current:int=0,
        device='cuda',
        act_scales=torch.exp,
        act_quats=F.normalize,
        act_colors=torch.sigmoid,
        act_opacities=torch.sigmoid,
        block_width:int=16,
        # The following are set by src/args.py:TrainParams
        lr_positions:float=0.00016,
        lr_scales:float=0.005,
        lr_quats:float=0.001,
        lr_colors:float=0.0025,
        lr_opacities:float=0.05,
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

        - `colors:Optional[torch.Tensor]` - Use this pre-defined parameter to initialize Gaussians. Shape `N,3,D`.
        Where `D=(sh_degree+1)**2`.
        If not specified, initialize uniformly in `0,1` (note that this will be passed through a sigmoid before rendering).

        - `opacities:Optional[torch.Tensor]` - Use this pre-defined parameter to initialize Gaussians. Shape `N,1`.
        If not specified, initialize to 1 (note that this will be passed through a sigmoid before rendering).

        - `scene_size:float` - Scale of the scene, used to initialize means if unspecified.

        - `sh_degree:int` - Maximum SH degree for color feature representation. Default is 0, which gives a 1-dimensional
        feature for RGB, which is equivalent to just no fancy features and direct RGB.
        Note that model will initialize with `Gaussians.sh_current=0`, to learn the 0-degree first. Can be upped using
        `Gaussians.oneup-sh()`.

        - `sh_current:int` - Number of SH degrees to use from init, default is 0.

        - `act_scales`,`act_quats`,`act_colors`,`act_opacities` - Activation functions, applied before rendering

        - `lr_position`,`lr_scales`,`lr_quats`,`lr_colors`,`lr_opacities` - Learning rates for optimizer.
        Note that `lr_position` will be scaled by `scene_extent`.

        `N` is the number of splats (either `num_points` or `means.shape[0]`)
        """
        # Init `torch.nn.Module` instance
        super().__init__()

        self.device = device

        # Use pre-defined means
        if means is not None:
            num_points = means.shape[0]

        # Use pre-defined number of points
        elif num_points is not None:
            means = scene_extend * 2*( torch.rand(num_points, 3, device=self.device) - .5 )

        # Raise error when neither means nor num_points is set
        else:
            raise ValueError("Either set `means` or `num_points` when initializing `Gaussians` instance")

        # Initialize scales
        if scales is None:
            scales = torch.tensor( knn_scale(means.detach().cpu().numpy()), device=self.device, dtype=torch.float32 )

        # Initialize rotation
        if quats is None:
            # From: https://github.com/nerfstudio-project/gsplat/blob/main/examples/simple_trainer.py
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
        self.sh_degree_max = sh_degree
        self.sh_degree_current = sh_current
        if colors is None:
            D = (sh_degree+1)**2
            # Initialize non-zero bands to zero
            colors = torch.zeros(num_points, D, 3, device=self.device)
            # Initialize 0-degree to random
            colors[:,0,:] = torch.rand(num_points, 3, device=self.device)

        # Sanity check
        if colors.shape[1] != (self.sh_degree_max+1)**2:
            raise ValueError("Colors have wrong shape, expecting N,D,C with D=(sh_degree+1)^2. "
                             f"Got {colors.shape} with sh_degree {sh_degree}")

        # Initialize opacities
        if opacities is None:
            # such that sigmoid(opacity) = 0.1
            opacities = torch.logit( torch.ones((num_points, 1), device=self.device) * 0.1 )

        # Sanity check: make sure all parameters have same number of points
        if not all(param.shape[0] == num_points for param in (means,scales,quats,colors,opacities)):
            raise ValueError("Not all parameters of Gaussians have the same first dimension")

        # Save as Parameters
        self.means = nn.Parameter(means.float())
        self.scales = nn.Parameter(scales.float())
        self.quats = nn.Parameter(quats.float())
        self.colors_dc = nn.Parameter(colors[:,:1,:].float())
        self.colors_fc = nn.Parameter(colors[:,1:,:].float())
        self.opacities = nn.Parameter(opacities.float())

        # Set up optimizer
        self.optimizer = optim.Adam(
            [
                {'params': [self.means], 'lr':lr_positions*scene_extend, 'name': 'means'},
                {'params': [self.scales], 'lr':lr_scales, 'name': 'scales'},
                {'params': [self.quats], 'lr':lr_quats, 'name': 'quats'},
                {'params': [self.colors_dc], 'lr':lr_colors, 'name': 'colors_dc'},
                {'params': [self.colors_fc], 'lr':lr_colors/20, 'name': 'colors_fc'},
                {'params': [self.opacities], 'lr':lr_opacities, 'name': 'opacities'},
            ],
            eps=1e-15,
            lr=0.0,
        )

        # Can be optionally set by `init_scheduler`
        self.lr_schedule = None

        # Tile size when rendering
        self.block_width = block_width

        # Set activation functions
        self.act_scales = act_scales
        self.act_quats = act_quats
        self.act_colors = act_colors
        self.act_opacities = act_opacities

        # Reset gradient stats, make sure no NameError happens
        self.reset_densification_stats()

        # Keep a running tally of number of split,cloned and pruned points for debugging
        self._n_split = 0
        self._n_clone = 0
        self._n_prune = 0
        self._n_prune_opacity = 0
        self._n_prune_radii = 0
        self._n_prune_scale = 0


    def init_lr_schedule(self, warmup_until=400, decay_from=15_000, decay_for=15_000):
        # Default learning rate schedule: 100 warmup iters, decay from there
        warmup = lr_utils.cosine_warmup(warmup_until, start=1e-8, end=1)
        decay = lr_utils.log_linear(decay_for, start=1, end=1e-2)

        # Set scheduler for each group
        lambdas = []
        for group in self.optimizer.param_groups:
            lmbda = lambda _: 1.
            if group["name"] in {"means", "colors", "scales"}:
                lmbda = lambda epoch: \
                    warmup(epoch) if epoch <= warmup_until \
                    else decay(epoch-decay_from) if epoch >= decay_from \
                    else 1.
            else:
                lmbda = lambda epoch: \
                    warmup(epoch) if epoch <= warmup_until \
                    else decay(epoch-decay_from) if epoch >= decay_from \
                    else 1.
            lambdas.append(lmbda)

        # Create actualy lr scheduler
        self.lr_schedule = optim.lr_scheduler.LambdaLR(
            optimizer=self.optimizer,
            lr_lambda=lambdas,
        )


    @property
    def colors(self) -> torch.Tensor:
        return torch.hstack((self.colors_dc, self.colors_fc))


    @property
    def num_points(self) -> int:
        return self.means.shape[0]


    def render(
            self,
            camera:Camera,
            glob_scale:float=1.0,
            bg:torch.Tensor=None,
            blur:bool=False # TODO
        ) -> RenderPackage:
        """
        Render a Camera instance using gaussian splatting

        For info on the gsplat library, see
        https://docs.gsplat.studio/

        Parameters
        - `camera:Camera` - Camera to render
        - `glob_scale:float=1.0` - Global scaling factor for the gaussians
        - `bg:Optional[torch.Tensor]=None` - Background to use, random if None

        Returns a RenderPackage instance with properties
        - `rendering:torch.Tensor` - Generated image, shape `C,H,W` where `C` is
        equal to `Gaussians.colors.shape[-1]` and `H,W` is determined by the given
        `Camera` instance.
        - `xys:torch.Tensor` - 2D location of splats, shape `N,2` where `N` is
        equal to `Gaussians.num_points`. Will attempt to retain grads, can be
        used for densification stats.
        - `visibility_mask:torch.Tensor` - Mask of visible splats, shape `N`
        - `alpha:torch.Tensor` - Alpha value of each pixel, shape `H,W`
        """
        # If unspecified generate random background
        if bg is None:
            bg = torch.rand(3, device=self.device)

        # Project Gaussians from 3D to 2D
        xys, depths, radii, conics, compensation, num_tiles_hit, cov3d = gsplat.project_gaussians(
            means3d=self.means,
            scales=self.act_scales( self.scales ),
            glob_scale=glob_scale,
            quats=self.act_quats( self.quats ),
            viewmat=camera.viewmat[:3,:],
            projmat=camera.projmat @ camera.viewmat,
            fx=camera.fx,
            fy=camera.fy,
            cx=camera.cx,
            cy=camera.cy,
            img_width=camera.W,
            img_height=camera.H,
            block_width=self.block_width,
            clip_thresh=camera.znear,
        )

        # Attempt to keep position gradients to update densification stats
        try:
            xys.retain_grad()
        except:
            pass

        if self.sh_degree_max > 0:
            viewdirs = self.means.detach() - torch.from_numpy(camera.loc).to(device=self.device, dtype=torch.float32).unsqueeze(0)
            viewdirs = viewdirs / viewdirs.norm(dim=-1, keepdim=True)

            # Convert SH features to features
            colors = gsplat.spherical_harmonics(
                degrees_to_use=self.sh_degree_current,
                viewdirs=viewdirs,
                coeffs=self.colors,
            )

            colors = torch.clamp(colors + 0.5, min=0.0)
        else:
            colors = self.act_colors( self.colors.view(self.num_points,-1) )


        # Generate image
        out_img, out_alpha = gsplat.rasterize_gaussians(
            xys=xys,
            depths=depths,
            radii=radii,
            conics=conics,
            num_tiles_hit=num_tiles_hit,
            colors=colors,
            opacity=self.act_opacities( self.opacities ) * compensation.unsqueeze(-1),
            img_height=camera.H,
            img_width=camera.W,
            block_width=self.block_width,
            background=bg,
            return_alpha=True,
        )

        rendering = out_img.permute(2,0,1)
        rendering = torch.clamp(rendering, min=0.0, max=1.0)

        return RenderPackage(
            rendering=rendering,
            xys=xys,
            radii=radii,
            visibility_mask=(radii > 0).squeeze(),
            alpha=out_alpha,
        )
    forward=render


    def reset_densification_stats(self) -> None:
        """
        Reset densification stats, automatically called after densifying
        """
        # Reset running average
        self._xys_grad_accum = torch.zeros((self.num_points,), device=self.device)
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

        Make sure all arguments are on the same device
        """

        # Update running average of visible Gaussians
        if xys.grad is not None:
            grads = xys.grad.detach().norm(dim=-1)
            self._xys_grad_accum[visibility_mask] += grads[visibility_mask]
            self._xys_grad_norm[visibility_mask] += 1

            # print(f"grad norm min {grads.min().item()} max {grads.max().item()} mean {grads.mean().item()} size {grads.shape}")

        # Update max radii
        self._max_radii[visibility_mask] = torch.maximum(
            self._max_radii[visibility_mask],
            radii.detach()[visibility_mask],
        )


    def densify(
            self,
            grad_threshold:float,
            max_density:float,
            min_opacity:float,
            max_world_size:float,
            max_screen_size:float=None,
            N:int=2
        ):
        """
        Clone and Split Gaussians based on stats acumulated with rendering. Select
        Gaussians with a high viewspace gradient for cloning and splitting

        Clone: Increase volume and number of Gaussians in underconstructed areas
        by duplicating splats.

        Split: Increase number of Gaussians, but keep the total volume the same
        by replacing one splat by two smaller ones, based on the scale of the original.

        For every splat selected for splitting, `N` new means are sampled using
        the original splat as a PDF. Opacity, color and rotation are duplicated for these
        new splats, and the scale is reduced by a factor `1/(.8*N)` for splats that are
        selected to be split (cloned splats will have the same scale as the original).

        Aditionally, discard any nearly transparent splat, as determined by `min_opacity`.
        Also discard any splat that was rendered with a radius larger than `max_radius` (if this
        arg is not None)

        Parameters:
        - `grad_threshold:float` - If the norm of the 2D spatial gradient of a splat exceeds
        this threshold, select it for splitting / cloning.
        - `max_density:float` - If the scale of a splat exceeds this threshold, select it for
        splitting.
        - `min_opacity:float` - Remove any splat with a opacity lower than this value.
        - `max_world_size:float` - If the scale of a splat exceeds this threshold, select it for
        pruning.
        - `max_screen_size:Optional[float]=None` - Remove any splat that was rendered with a
        radius larger than this value. If None, skip this step.
        - `N:int=2` - Into how many splats a splat needs to be split
        """
        # Compute actual gradient average
        grads = self._xys_grad_accum / self._xys_grad_norm
        grads[grads.isnan()] = 0.

        grad_cond = grads >= grad_threshold
        scale_cond = ( torch.max(self.act_scales(self.scales), dim=-1).values > max_density ).squeeze()

        # Clone all splats with a big gradient but not too big a scale
        clone_cond =  grad_cond & ( ~scale_cond )

        self._n_clone = clone_cond.sum()

        # Create copy of each selected splat
        means_clone = self.means[clone_cond].detach().clone()
        scales_clone = self.scales[clone_cond].detach().clone()
        quats_clone = self.quats[clone_cond].detach().clone()
        colors_clone = self.colors[clone_cond].detach().clone()
        opacities_clone = self.opacities[clone_cond].detach().clone()

        # Split all splats with a big gradient and a large scale
        split_cond =  grad_cond & scale_cond

        self._n_split = split_cond.sum()

        # Create `N` copies of each split splat
        scales_split = self.scales[split_cond].repeat( N, 1 )
        quats_split = self.quats[split_cond].repeat( N, 1 )
        colors_split = self.colors[split_cond].repeat( N, 1, 1 )
        opacities_split = self.opacities[split_cond].repeat( N, 1 )

        # Downscale these splats (assumes that Gaussians.scales_act is torch.exp)
        scales_split = torch.log( torch.exp( scales_split ) / (.8*N) )

        # Add a noisy vector to the mean based on the scale and rotation of the splat
        noise = torch.normal(
            mean=torch.zeros((scales_split.shape[0],3), device=self.device),
            std=self.act_scales(scales_split),
        )
        rotmats = qvec2rotmat(self.act_quats(quats_split.detach()))

        # Get mean as sum of noise and original mean
        means_split = self.means[split_cond].repeat( N, 1 ) + torch.bmm(rotmats, noise.unsqueeze(-1)).squeeze(-1)

        # Prune any point with an opacity too low
        prune_cond = (self.act_opacities( self.opacities ) <= min_opacity).squeeze()
        self._n_prune_opacity = prune_cond.sum()

        # If set, discard any splat that was rendered with a view radius larger than a max radius
        if max_screen_size is not None:

            # Prune any point with a screen radius larger than this
            prune_radii_cond = (self._max_radii > max_screen_size)
            self._n_prune_radii = prune_radii_cond.sum()

            # Prune any point with a scale larger than a world max
            prune_scale_cond = ( torch.max(self.act_scales(self.scales), dim=1).values > max_world_size ).squeeze()
            self._n_prune_scale = prune_scale_cond.sum()

            # Combine conditions
            prune_cond = prune_cond | prune_radii_cond | prune_scale_cond

        # Total number of points pruned
        prune_cond = prune_cond | split_cond
        self._n_prune = prune_cond.sum()

        # Add the unselected and split/pruned points together
        self.update_optimizer(
            dict_to_cat={
                'means':     torch.cat( (    means_clone,     means_split), dim=0),
                'scales':    torch.cat( (   scales_clone,    scales_split), dim=0),
                'quats':     torch.cat( (    quats_clone,     quats_split), dim=0),
                'colors_dc': torch.cat( (   colors_clone[:,:1,:],    colors_split[:,:1,:]), dim=0),
                'colors_fc': torch.cat( (   colors_clone[:,1:,:],    colors_split[:,1:,:]), dim=0),
                'opacities': torch.cat( (opacities_clone, opacities_split), dim=0),
            },
            prune_mask=prune_cond
        )

        self.reset_densification_stats()


    def update_optimizer(
            self,
            dict_to_cat:dict[str,torch.Tensor],
            prune_mask:torch.BoolTensor,
        ) -> None:
        """
        Concatenate and prune existing parameters in optimizer

        This method assumes each parameter group is associated with 1 Parameter instance
        ie every `group` in `Gaussians.optimizer.param_groups` satisfies `len(group['params']) == 1`

        - `dict_to_cat:dict[str,tuple[torch.Tensor]]` A dictionary mapping the name of a parameter group
        to a tensor to concatenate to that group
        - `prune_mask:torch.BoolTensor` A mask or slice of what items of the original group to keep

        From https://github.com/graphdeco-inria/gaussian-splatting/blob/2eee0e26d2d5fd00ec462df47752223952f6bf4e/scene/gaussian_model.py#L307
        """


        keep_mask = ~prune_mask

        # Iterate parameter groups
        for group in self.optimizer.param_groups:

            if group['name'] not in dict_to_cat: continue;

            # Get tensor to concatenate, or empty if nothing to add
            extension_tensor = dict_to_cat.get(group["name"], torch.empty(0))

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


    def reset_opacity(self, value=0.01):
        """
        Clip all opacities to a fixed maximum
        """
        # clamp_max_ is inplace
        self.opacities.data.clamp_(max=torch.logit(torch.tensor(value, device=self.device)))

        # Remove state
        for group in self.optimizer.param_groups:
            if group['name'] == 'opacities':
                param = group["params"][0]
                param_state = self.optimizer.state[param]
                param_state["exp_avg"] = torch.zeros_like(param_state["exp_avg"])
                param_state["exp_avg_sq"] = torch.zeros_like(param_state["exp_avg_sq"])

                del self.optimizer.state[param]
                self.optimizer.state[param] = param_state


    def oneup_sh_degree(self):
        """
        Increment `Gaussians.sh_degree_current`,
        capped at `Gaussians.sh_degree_max`
        """
        self.sh_degree_current = min(
            self.sh_degree_max,
            self.sh_degree_current+1
        )
