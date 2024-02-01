from collections import namedtuple
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

# An instance of this is returned by the `Gaussians.render` method
RenderPackage = namedtuple("RenderPackage", ["rendering","xys","radii","visibility_mask"])

class Gaussians(nn.Module):
    BLOCK_X, BLOCK_Y = 16, 16


    @classmethod
    def from_colmap(
        cls,
        path_to_points3D_file:str,
        device='cuda',
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
        device='cuda',
        **kwargs
    ) -> 'Gaussians':
        """
        Read `.ply` file and initialize Gaussians using positions and colors
        from PlyData.

        Any additional kwargs will be passed to Gaussians.__init__
        """
        # Read ply data
        plydata = plyfile.PlyData.read(path_to_ply_file)

        # Pass this to the __init__ function
        params = {}

        # Extract positions as np array
        vertex:plyfile.PlyElement = plydata['vertex']

        # Get properties as a set of strings for easy subset checking
        properties = {prop.name for prop in vertex.properties}

        # convert either xyz or pos_1/2/3 to means
        if properties.issuperset('xyz'):
            means = np.stack([vertex['x'], vertex['y'], vertex['z']], axis=1)
        elif properties.issuperset({'pos_1','pos_2','pos_3'}):
            means = np.stack([vertex['pos_1'], vertex['pos_2'], vertex['pos_3']], axis=1)
        else:
            raise ValueError("Cannot read .ply data, vertex needs either xyz or pos_1/2/3 properties")
        params['means'] = torch.from_numpy(means).to(device)

        # Get scales
        if properties.issuperset({"scale_1","scale_2","scale_3"}):
            scales = np.stack([vertex['scale_1'], vertex['scale_2'], vertex['scale_3']], axis=1)
            params['scales'] = torch.from_numpy(scales).to(device)

        # Convert rot_1/2/3/4 to quaternions
        if properties.issuperset({"rot_1","rot_2","rot_3","rot_4"}):
            quats = np.stack([vertex['rot_1'], vertex['rot_2'], vertex['rot_3'], vertex['rot_4']], axis=1)
            params['quats'] = torch.from_numpy(quats).to(device)

        # Extract colors
        colors = None
        if properties.issuperset({'red','green','blue'}):
            colors = np.stack([vertex['red'], vertex['green'], vertex['blue']], axis=1) / 255.0
            params['colors'] = torch.from_numpy(colors).to(device).reshape(-1, 1, 3)
            params['sh_degree']=0

        # Convert color_1,color_2,...,color_C to NxC numpy array
        elif 'color_1' in properties:
            cols = sorted([prop for prop in properties if prop.startswith("color_")])
            colors = np.stack([vertex[c] for c in cols], axis=1)
            params['colors'] = torch.from_numpy(colors).to(device).reshape(-1, 1, len(cols))
            params['sh_degree']=0

        # Convert color_1_1, ..., color_D_1, ..., color_D_C to NxDxC numpy array
        elif 'color_1_1' in properties:
            colors = []

            i=1
            while f'color_{i}_1' in properties:
                # Get color_i_1, ..., color_i_C
                cols = sorted([prop for prop in properties if prop.startswith(f"color_{i}")])
                colors.append( np.stack([vertex[c] for c in cols],axis=1) )
                i+=1

            # Stack colors together. Assumes all stacked columns have the same shape
            colors = np.stack(colors,axis=2)
            D = colors.shape[-2]
            sh_degree:float = np.sqrt(D)-1

            # Sanity check, make sure D=(d+1)^2 for some sh degree d
            if not sh_degree.is_integer():
                raise ValueError(f"Unexpected number of features per color, "
                                 "got D={D}, expected D=(d+1)^2 for some degree d.")

            params['sh_degree'] = sh_degree
            params['colors'] = torch.from_numpy( colors ).to(device)

        # Extract opacities
        if 'opacity' in properties:
            params['opacities'] = torch.from_numpy( vertex['opacity'] ).to(device)

        # Create Gaussians instance
        return Gaussians(
            device=device,
            **params,
            **kwargs,
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
        sh_degree:int=3,
        sh_current:int=0,
        device='cuda',
        act_scales=torch.exp,
        act_quats=F.normalize,
        act_colors=torch.sigmoid,
        act_opacities=torch.sigmoid,
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
            scales = torch.log( torch.rand(num_points, 3, device=self.device) / (scene_extend*20) )

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
            raise ValueError("Colors have wrong shape, expecting N,3,D with D=(d+1)^2 for some sh degree d. "
                             f"Got {colors.shape}")

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
                {'params': [self.means], 'lr':lr_positions*scene_extend, 'name': 'means'},
                {'params': [self.scales], 'lr':lr_scales, 'name': 'scales'},
                {'params': [self.quats], 'lr':lr_quats, 'name': 'quats'},
                {'params': [self.colors], 'lr':lr_colors, 'name': 'colors'},
                {'params': [self.opacities], 'lr':lr_opacities, 'name': 'opacities'},
            ],
            eps=1e-15,
            lr=0.0,
        )

        # Set activation functions
        self.act_scales = act_scales
        self.act_quats = act_quats
        self.act_colors = act_colors
        self.act_opacities = act_opacities

        # Reset gradient stats, make sure no NameError happens
        self.reset_densification_stats()

        # TODO: remove these
        self._n_split = 0
        self._n_clone = 0
        self._n_prune = 0


    @property
    def num_points(self) -> int:
        return self.means.shape[0]


    def render(
            self,
            camera:Camera,
            glob_scale:float=1.0,
            bg:torch.Tensor=None,
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
        - `rendering:torch.Tensor` - Generated image, shape `H,W,C` where `C` is
        equal to `Gaussians.colors.shape[-1]` and `H,W` is determined by the given
        `Camera` instance.
        - `xys:torch.Tensor` - 2D location of splats, shape `N,2` where `N` is
        equal to `Gaussians.num_points`. Will attempt to retain grads, can be
        used for densification stats.
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

        # TODO: remove this if CUDA errors stop
        # Sanity check, render nothing if nothing is in view
        # print((radii>0).sum().item(),flush=True)
        if not (radii > 0).any():
            raise ValueError("No splats in view")

        torch.cuda.synchronize()

        # Attempt to keep position gradients to update densification stats
        try:
            xys.retain_grad()
        except:
            pass

        # TODO
        viewdirs = self.means - torch.from_numpy(camera.t).to(self.device).unsqueeze(0)
        viewdirs /= torch.linalg.norm(viewdirs,dim=1,keepdim=True)

        # Convert SH features to features
        colors = gsplat.spherical_harmonics(
            degrees_to_use=self.sh_degree_current,
            viewdirs=viewdirs,
            coeffs=self.colors,
        )

        # Generate image
        out_img = gsplat.rasterize_gaussians(
            xys=xys,
            depths=depths,
            radii=radii,
            conics=conics,
            num_tiles_hit=num_tiles_hit,
            colors=self.act_colors( colors ),
            opacity=self.act_opacities( self.opacities ),
            img_height=camera.H,
            img_width=camera.W,
            background=bg,
        )

        return RenderPackage(
            rendering=out_img,
            xys=xys,
            radii=radii,
            visibility_mask=(radii > 0).squeeze()
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
            self._xys_grad_accum[visibility_mask] += torch.linalg.norm(xys.grad[visibility_mask], dim=1)
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
        # print("#"*20,f"\nDensifying... [started with {self.num_points} splats]")

        # Compute actual gradient average
        grads = self._xys_grad_accum / self._xys_grad_norm
        grads[grads.isnan()] = 0.

        grad_cond = grads >= grad_threshold
        scale_cond = ( torch.max(self.act_scales(self.scales),axis=1).values > max_density ).squeeze()

        # Clone all splats with a big gradient but not too big a scale
        clone_cond =  grad_cond & ( ~scale_cond )

        # print(f"{clone_mask.sum()} splats selected for cloning/splitting")
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
        means_split = self.means[split_cond].repeat( N, 1 )
        scales_split = self.scales[split_cond].repeat( N, 1 )
        quats_split = self.quats[split_cond].repeat( N, 1 )
        colors_split = self.colors[split_cond].repeat( N, 1 )
        opacities_split = self.opacities[split_cond].repeat( N, 1 )

        # Downscale these splats (assumes that Gaussians.scales_act is torch.exp)
        scales_split = torch.log( torch.exp( scales_split ) / (.8*N) )

        # Add a noisy vector to the mean based on the scale and rotation of the splat
        noise = torch.normal(
            mean=torch.zeros_like(means_split, device=self.device),
            std=self.act_scales(scales_split),
        )
        rotmats = batch_qvec2rotmat(self.act_quats(quats_split))
        means_split += torch.bmm(rotmats, noise.unsqueeze(-1)).squeeze(-1)

        # Finally, remove all split splats, and with an opacity too low
        prune_cond = split_cond \
            | (self.act_opacities( self.opacities ) <= min_opacity).squeeze()

        # If set, discard any splat that was rendered with a view radius larger than a max radius
        if max_screen_size is not None:
            prune_cond = prune_cond \
                | (self._max_radii > max_screen_size) \
                | ( torch.max(self.act_scales(self.scales),axis=1).values > max_world_size ).squeeze()

        self._n_prune = prune_cond.sum()

        # Add the unselected and split/pruned points together
        self.update_optimizer(
            dict_to_cat={
                'means': torch.cat( (means_clone, means_split), dim=0),
                'scales': torch.cat( (scales_clone, scales_split), dim=0),
                'quats': torch.cat( (quats_clone, quats_split), dim=0),
                'colors': torch.cat( (colors_clone, colors_split), dim=0),
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
        # clamp_max_ is inplace
        self.opacities.clamp_max_(-4.59512)
        del self.optimizer.state[self.opacities]
