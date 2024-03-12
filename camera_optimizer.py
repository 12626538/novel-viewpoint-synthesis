import os
import numpy as np

try:
    from tqdm import tqdm
    USE_TQDM=True
except ModuleNotFoundError:
    USE_TQDM=False

import torch
from torchvision.utils import save_image
from torcheval.metrics.functional import peak_signal_noise_ratio as psnr


try:
    from torch.utils.tensorboard import SummaryWriter
    USE_TENSORBOARD=True
except ModuleNotFoundError:
    print("Warning! Tensorboard not available")
    USE_TENSORBOARD=False

from src.model.gaussians import Gaussians,RenderPackage
from src.data import DataSet
from src.utils.metric_utils import get_loss_fn,SSIM,LPIPS
from src.arg import ModelParams,DataParams,TrainParams,PipeLineParams,get_args

from src.camera.camp import UnnamedCameraType, CameraParameterization, DCCamera
import random

def render_params(params:list[CameraParameterization],model:Gaussians,name:str):

    background = torch.zeros(3, device=model.device)
    for idx,param in enumerate(params):
        camera = UnnamedCameraType.get_camera(param)
        rendering_pkg = model.render(camera, bg=background)
        save_image(rendering_pkg.rendering,f"renders/camoptim/cam_{idx:06d}_{name}.png")

def save_params(params:DataSet):
    pass


def train_loop(
    model:Gaussians,
    dataset:DataSet,
    device,
    train_args:TrainParams,
    pipeline_args:PipeLineParams
):

    # Set up loss
    loss_fn = get_loss_fn(vars(train_args))

    # Keep a running (smoothed) loss for logging
    loss_smooth_mult = .9
    loss_smooth = 0.

    # Set background
    background = None
    if not pipeline_args.random_background:
        background = torch.tensor([1,1,1] if pipeline_args.white_background else [0,0,0], dtype=torch.float32, device=device)

    pbar = None
    if USE_TQDM:
        pbar = tqdm(total=train_args.iterations, ncols=80)

    optimizer = torch.optim.Adam(
        params=[{'name':'cameras', 'params':[cam.latent for cam in params]}],
        lr=1e-4
    )

    losses = []

    dataset_cycler = dataset.iter("train",cycle=True,shuffle=True)
    for iter in range(1,train_args.iterations+1):
        optimizer.zero_grad()

        # Forward pass
        param = next(dataset_cycler)
        camera = UnnamedCameraType.get_camera(param)
        rendering_pkg = model.render(camera, bg=background, blur=pipeline_args.do_blur)

        loss,loss_pkg = loss_fn(rendering_pkg.rendering, camera.gt_image)

        # Compute gradients
        loss.backward()

        optimizer.step()

        # Keep a smoothed loss
        if loss_smooth == 0:
            loss_smooth = loss.item()
        else:
            loss_smooth = loss_smooth_mult * loss_smooth \
                + (1-loss_smooth_mult) * loss.item()

        losses.append(loss_smooth)

        if iter%100 == 0 and pbar is not None:
            pbar.update(100)


    if pbar is not None:
        pbar.close()

    return losses

if __name__ == '__main__':

    # Get args
    args,(data_args,model_args,train_args,pipeline_args) = get_args(
        DataParams, ModelParams, TrainParams, PipeLineParams,
        save_args=False
    )

    if pipeline_args.no_pbar:
        USE_TQDM = False

    # Set torch device
    try:
        torch.cuda.set_device(args.device)
    except:
        print(f"!Warning! Could not use device {args.device}, falling back to cuda:0.")
        args.device = 'cuda:0'

    print("Reading from 3DU dataset...")
    dataset = DataSet.from_3du(
        device=args.device,
        **vars(data_args)
    )

    print("Initializing camera parameterizations...")
    params = DataSet([
        UnnamedCameraType.parameterize(cam) for cam in dataset.cameras
    ])

    # Use checkpoint
    if os.path.isfile( pipeline_args.load_checkpoint):
        print(f"Loading .ply model from {pipeline_args.load_checkpoint}")
        model = Gaussians.from_ply(
            device=args.device,
            path_to_ply_file=pipeline_args.load_checkpoint,
            **vars(model_args)
        )

    # Use 3DU directory to find a `point_cloud.ply` file
    else:
        fname = os.path.join(data_args.source_path, 'point_cloud.ply')
        print(f"Loading .ply model from 3DU dataset directory {fname}")
        model = Gaussians.from_ply(
            device=args.device,
            path_to_ply_file=fname,
            **vars(model_args)
        )
    os.makedirs("renders/camoptim",exist_ok=True)
    render_params(params,model,"before")
    losses = train_loop(
        model=model,
        dataset=params,
        device=args.device,
        train_args=train_args,
        pipeline_args=pipeline_args
    )
    render_params(params,model,"after")

    save_params(params)
