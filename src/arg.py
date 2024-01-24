import os
import datetime
import argparse

parser = argparse.ArgumentParser(
    prog="Novel Viewpoint Synthesis using gplsat",
)

# DATASET ARGUMENTS ============================================================
dataset_group = parser.add_argument_group("Dataset arguments")
dataset_group.add_argument(
    "-s", "--source-path",
    required=True,
    help="Dataset to use, needs to contain `images` and `sparse/0/[images|cameras|points3D].txt` files",
)
dataset_group.add_argument(
    "--images-folder",
    default='images',
    help="Use `[source-dir]/[image_folder]` as image folder, allows for downsampled `images_2` subdir",
)
dataset_group.add_argument(
    "--meta-folder",
    default='sparse/0',
    help="Use `[source-dir]/[meta-fodler]` as [camera|image|points].txt data",
)

# TRAIN ARGUMENTS ==============================================================
train_group = parser.add_argument_group("Training arguments")
train_group.add_argument(
    "--num-iterations", type=int,
    default=7_000,
    help="Number of iterations to train",
)
train_group.add_argument(
    "--densify-from", type=int,
    default=5000,
)
train_group.add_argument(
    "--densify-until", type=int,
    default=6000,
)
train_group.add_argument(
    "--densify-every", type=int,
    default=100,
)
train_group.add_argument(
    "--save-at", type=int, nargs="+",
    default=[7_000,30_000],
    help="Save model checkpoint at these iterations"
)
train_group.add_argument(
    "--test-at", type=int, nargs="+",
    default=[7_000,30_000],
    help="Test model at these iterations"
)
train_group.add_argument(
    "--lr-position", type=float,
    default=0.0016,
)
train_group.add_argument(
    "--lr-scale", type=float,
    default=0.0016,
)
train_group.add_argument(
    "--lr-rot", type=float,
    default=0.0016,
)
train_group.add_argument(
    "--lr-color", type=float,
    default=0.0016,
)
train_group.add_argument(
    "--lr-opacitiy", type=float,
    default=0.0016,
)
train_group.add_argument(
    "--grad-treshold", type=float,
    default=1e-8,
    help="Grad threshold to select splats for cloning/splitting"
)
train_group.add_argument(
    "--max-density", type=float,
    default=0.01,
    help="Maximum splat density, will be scaled by scene extant"
)


# MODEL ARGUMENTS ==============================================================
model_group = parser.add_argument_group("Model arguments")
model_group.add_argument(
    "-c", "--checkpoint",
    default=None,
    help="Load from this checkpoint",
)
model_group.add_argument(
    "-m", "--model-dir",
    default=os.path.join("output", datetime.datetime.now().strftime('%a%d%b%H%M').lower()),
    help="Load from this checkpoint, default is `output/[current time]`",
)


# MISC ARGUMENTS ===============================================================
options_group = parser.add_argument_group("TensorBoard arguments")
options_group.add_argument(
    "--log-dir",
    default="./logs/",
    help="If tensorboard is installed, save results here"
)

args = parser.parse_args()
print(args)

# Split `args` back into groups
# From https://stackoverflow.com/a/46929320
arg_groups={}
for group in parser._action_groups:
    group_dict={a.dest:getattr(args,a.dest,None) for a in group._group_actions}
    arg_groups[group.title]=argparse.Namespace(**group_dict)
print(arg_groups)
