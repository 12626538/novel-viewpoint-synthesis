import os
import datetime
import configparser
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
                kwargs['action'] = 'store_false' if value else 'store_true'
            elif type(value) == list:
                kwargs['nargs'] = '+'
                kwargs['type'] = type(value[0])
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

        self.sh_degree = 2

        self.lr_positions = 0.00016
        self.lr_scales    = 0.005
        self.lr_quats     = 0.001
        self.lr_colors    = 0.0025
        self.lr_opacities = 0.05
        self.lr_blur      = 0.001
        super().__init__(*arg,**kwarg)


class DataParams(ParamGroup):
    """
    Arguments to be passed when construction a ColmapDataSet instance
    """
    def __init__(self, *arg,**kwarg):
        self._source_path=''
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

        self.grad_threshold=1e-6
        self.max_density=0.05
        self.min_opacity=0.005
        self.max_screen_size=0.15

        self.loss_weight_mse = 1.
        self.loss_weight_dssim = .2
        self.loss_weight_lpips = .1
        self.loss_weight_mae = 0.

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

        self.model_name = datetime.datetime.now().strftime('%b%d%a%H%M%S').lower()

        self.model_dir = 'models'
        self.load_checkpoint = ''

        # For tensorboard
        self.log_dir="models"

        self.random_background = False
        self.white_background = False

        self.no_pbar = False
        self.do_blur = False
        super().__init__(*arg,**kwarg)

    def extract(self,args):
        if args.load_checkpoint and not args.load_checkpoint.endswith(".ply"):
            args.load_checkpoint = os.path.join(
                os.path.abspath(args.load_checkpoint),
                "point_cloud.ply"
            )

        args.model_dir = os.path.join( os.path.abspath(args.model_dir), args.model_name)
        args.log_dir = os.path.join( os.path.abspath(args.log_dir), args.model_name)


        return super().extract(args)

def get_args(
        *groups:type[ParamGroup],
        parser=ArgumentParser("Training Novel Viewpoint Synthesis"),
        save_args:bool=False,
    ) -> tuple[Namespace, list[Namespace]]:
    """
    Get args for a specified collection of parameter groups

    This is a quick wrapper to deal with
    1) Setting up a ArgumentParser
    2) Adding some number of ParamGroups to it
    3) Parsing the parser
    4) Splitting the parsed arguments back into parameter groups

    Parameters:
    - `groups` - Any positional arguments should be ParamGroup classes
        (not instances!)
    - `parser` - Optionally, a pre-defined `ArgumentParser` can be passed
        as `parser` kwarg. This allows user to add arguments before passing it
        through this function
    - `save_args:bool=False` - If set to True, create a `args.ini` file inside
        the `args.model_dir` with all arguments in the argparser

    Returns:
    - `args:Namespace` - The args returned by the `parser`
    - For each `group` passed as input, a `Namespace` instance for that
        parameter group is returned.)

    Example:
    >>> args, (args_data,args_model) = get_args(DataParams, ModelParams)

    Is equivalent to
    >>> parser = ArgumentParser()
    >>> dp = DataParams(parser)
    >>> mp = ModelParams(parser)
    >>> args = parser.parse_args()
    >>> args_data = dp.extract(args)
    >>> args_model = mp.extract(args)
    """

    # Add parameter groups to parser
    parser_groups:list[ParamGroup] = []
    for group in groups:
        parser_groups.append( group(parser) )

    # Extract all args
    args = parser.parse_args()

    # Split `args` back into groups
    args_groups:list[Namespace] = []
    for parser_group in parser_groups:
        args_groups.append( parser_group.extract(args) )

    # Save args in model directory
    if save_args and hasattr(args,'model_dir'):
        config = configparser.ConfigParser()
        config['parameters'] = {}

        for key,value in vars(args).items():
            config['parameters'][key] = str(value)

        os.makedirs(args.model_dir,exist_ok=True)
        with open(os.path.join(args.model_dir,'args.ini'), 'w') as f:
            print("Saving args at", f.name)
            config.write(f)

    # Return `args` and all groups
    return args, args_groups
