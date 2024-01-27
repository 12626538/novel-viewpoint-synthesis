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

            # If set, replace any value equivalent to False to None
            # for example, replaces empty strings with None, which makes it a required argument
            kwargs['default'] = value

            group.add_argument(*name_or_flag, **kwargs)

    def extract(self, args) -> Namespace:
        group = Namespace()
        for var,val in vars(args).items():
            if var in vars(self) or ("_" + var) in vars(self):
                setattr(group, var, val)
        return group

class ModelParams(ParamGroup):
    def __init__(self, *arg,**kwarg):
        self.lr_position = 0.0016
        self.lr_scale    = 0.0016
        self.lr_rot      = 0.0016
        self.lr_color    = 0.0016
        self.lr_opacitiy = 0.0016
        super().__init__(*arg,**kwarg)


class DataParams(ParamGroup):
    def __init__(self, *arg,**kwarg):
        self._source_path_=''
        self.images_folder='images'
        self.data_folder='sparse/0'
        super().__init__(*arg,**kwarg)


class TrainParams(ParamGroup):
    def __init__(self, *arg,**kwarg):
        self.iterations=30_000
        self.save_at=[7_000,30_000]
        self.test_at=[7_000,30_000]

        self.densify_from=2_000
        self.densify_every=500
        self.densify_until=15_000

        self.grad_threshold=1e-8
        self.max_density=0.01
        super().__init__(*arg,**kwarg)


class PipeLineParams(ParamGroup):
    def __init__(self, *arg,**kwarg):
        self.device='cuda:0'
        self.rescale=1
        self.load_checkpoint=''
        self.save_checkpoint=''
        self.log_dir='./logs/'
        super().__init__(*arg,**kwarg)
