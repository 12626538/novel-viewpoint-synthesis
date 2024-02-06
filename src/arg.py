import os
import datetime
from argparse import ArgumentParser,Namespace

class ParamGroup(object):
    """
    From
    https://github.com/graphdeco-inria/gaussian-splatting/blob/2eee0e26d2d5fd00ec462df47752223952f6bf4e/arguments/__init__.py#L19
    """
    def __init__(self, parser: ArgumentParser, fill_none = False):
        super().__init__()

        group = parser.add_argument_group(self.__class__.__name__)
        for key, value in vars(self).items():

            kwargs = dict()

            # hyphens are nicer than underscores
            key = key.replace('_','-')

            # Any variable ending in '-' is required
            if key.endswith('-'):
                key = key[:-1]
                kwargs['required'] = True

            # Any variable starting with '_' can be shorthanded by the first letter
            if key.startswith('-'):
                name_or_flag = ['-'+key[1:2], '--'+key[1:]]
            else:
                name_or_flag = ['--'+key, ]

            # Set type
            if type(value) == bool:
                kwargs['action'] = 'store_true'
            else:
                kwargs['type'] = type(value)

            # Set default
            kwargs['default'] = value

            group.add_argument(*name_or_flag, **kwargs)

    def extract(self, args) -> Namespace:
        group = Namespace()

        # Leading and trailing underscores were used to indicate shorthand/required args
        # Strip those now to keep in the final representation
        attrs = {var.strip('_') for var in vars(self)}

        for var,val in vars(args).items():
            if var in attrs:
                setattr(group, var, val)

        return group

class ModelParams(ParamGroup):
    """
    Arguments to be passed when construction a Gaussians instance
    """
    def __init__(self, *arg,**kwarg):
        self.num_points = 200_000

        self.sh_degree = 3

        self.lr_positions = 0.00016
        self.lr_scales    = 0.005
        self.lr_quats     = 0.001
        self.lr_colors    = 0.0025
        self.lr_opacities = 0.05
        super().__init__(*arg,**kwarg)


class DataParams(ParamGroup):
    """
    Arguments to be passed when construction a ColmapDataSet instance
    """
    def __init__(self, *arg,**kwarg):
        self._source_path_=''
        self.images_folder='images'
        self.data_folder='sparse/0'
        self.rescale=1
        super().__init__(*arg,**kwarg)

    def extract(self,args):
        args.source_path = os.path.abspath(args.source_path)
        return super().extract(args)


class TrainParams(ParamGroup):
    """
    Arguments to be passed to `train_loop` in `train.py`
    """
    def __init__(self, *arg,**kwarg):
        self.iterations=30_000
        self.save_at=[7_000,30_000]
        self.test_at=[7_000,30_000]

        self.densify_from=500
        self.densify_every=100
        self.densify_until=15_000

        self.reset_opacity_from=3_000
        self.reset_opacity_until=15_000
        self.reset_opacity_every=3_000

        self.oneup_sh_every=2_000

        self.grad_threshold=2e-6
        self.max_density=0.01
        self.min_opacity=0.005

        self.lambda_dssim=0.2

        super().__init__(*arg,**kwarg)

    def extract(self,args):
        args.save_at += [args.iterations,]
        args.test_at += [args.iterations,]

        return super().extract(args)


class PipeLineParams(ParamGroup):
    """
    Additional arguments used in `train.py`
    """
    def __init__(self, *arg,**kwarg):
        self.device='cuda:0'

        self.model_name = datetime.datetime.now().strftime('%a%d%b%H%M%S').lower()

        self.model_dir = 'models'
        self.load_checkpoint = ''

        # For tensorboard
        self.log_dir="logs/{}".format(self.model_name)

        self.no_pbar = False
        super().__init__(*arg,**kwarg)

    def extract(self,args):
        if args.load_checkpoint and not args.load_checkpoint.endswith(".ply"):
            args.load_checkpoint = os.path.join(
                os.path.abspath(args.load_checkpoint),
                "point_cloud.ply"
            )

        args.model_dir = os.path.join( os.path.abspath(args.model_dir), args.model_name)
        args.log_dir = os.path.abspath(args.log_dir)

        return super().extract(args)
