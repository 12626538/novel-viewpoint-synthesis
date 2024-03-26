import os
import numpy as np

try:
    import tqdm
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

import cv2

from src.model.gaussians import Gaussians,RenderPackage
from src.model.m2f import SegmentationModel
from src.data import DataSet
from src.arg import ModelParams,DataParams,TrainParams,PipeLineParams,get_args
from src.utils.metric_utils import get_loss_fn,SSIM,LPIPS
from src.utils import image_path_to_tensor


if __name__ == '__main__':

    # Get args
    args,(data_args,model_args,train_args,pipeline_args) = get_args(
        DataParams, ModelParams, TrainParams, PipeLineParams,
        save_args=False
    )

    # Set torch device
    try:
        torch.cuda.set_device(args.device)
    except:
        print(f"!Warning! Could not use device {args.device}, falling back to cuda:0.")
        args.device = 'cuda:0'

    # 3DU datasets from src/preprocessing/convert_3du.py
    if "3du_data_" in data_args.source_path:
        print("Reading from 3DU dataset...")
        dataset = DataSet.from_3du(
            device=args.device,
            **vars(data_args)
        )

    # Paper datasets are COLMAP formatted
    else:
        print("Reading dataset from COLMAP...")
        dataset = DataSet.from_colmap(device=args.device, **vars(data_args))

    # Read classes metadata
    class_names = SegmentationModel.CLASSES
    class_colors = torch.from_numpy( SegmentationModel.PALETTE / 255.).to(device=args.device)
    num_classes = len(class_names)


    # Read segmentation masks
    seg_dir = os.path.join(args.source_path,'segmentations')
    for cam in dataset.cameras:
        # TODO: make this more reliable
        path = os.path.join(seg_dir, cam.name.replace('.png','.npy'))

        # Read as shape C,H,W, convert to shape H*W,C
        segmentation = np.load(path)
        cam.gt_image = torch.from_numpy( segmentation.flatten() ).to(device=args.device, dtype=torch.long)

    # Use checkpoint
    if pipeline_args.load_checkpoint:
        print(f"Loading .ply model from {pipeline_args.load_checkpoint}")
        model = Gaussians.from_ply(
            device=args.device,
            path_to_ply_file=pipeline_args.load_checkpoint,
            **vars(model_args),
            num_additional_features=num_classes,
        )
        model.sh_degree_current = model.sh_degree_max
    else:
        raise ValueError("Please provide a checkpoint")

    os.makedirs('renders/segmentations/', exist_ok=True)

    loss_fn = torch.nn.CrossEntropyLoss(reduction='none')

    # Make sure there is at least one image to train on
    # assert any(cam.name in masks for cam in dataset.iter('train')), "There are no masks found to train on"

    torch.cuda.empty_cache()

    # Only train 'additional_features'
    for group in model.optimizer.param_groups:
        if group['name'] == 'additional_features':
            group['lr'] = 0.0010
        else:
            group['lr'] = 0.

    try:
        unk_class = class_names.index('unknown')
    except ValueError:
        unk_class = 0
    bg = torch.zeros((3+num_classes,), device='cuda', dtype=torch.float32)
    bg[3+unk_class] = 1.

    test_cam = dataset.cameras[ len(dataset.cameras)//2 ]

    # Set up train loop
    data_cycle = dataset.iter("train", cycle=True, shuffle=True)
    losses = []
    iters = 5_000
    pbar = range(iters) if not USE_TQDM else tqdm.trange(iters)
    for it in pbar:

        # Reset grad
        model.optimizer.zero_grad(set_to_none=True)

        # Get next cam to train on
        cam = next(data_cycle)

        # Forward pass
        pkg = model.render(cam, bg=bg)
        feat = pkg.additional_features.view(-1,num_classes)
        loss = loss_fn(feat, cam.gt_image)

        # Use alpha channel to account for areas with no splats
        loss = loss * pkg.alpha.flatten().detach()
        loss = loss.mean()

        # Backward step
        loss.backward()
        model.optimizer.step()

        # Iter report
        losses.append(loss.item())

        if it%50 == 0 or it == iters-1:
            pkg = model.render(test_cam, bg=bg)
            feat = pkg.additional_features.argmax(dim=-1)

            feat_rgb = class_colors[feat % class_colors.shape[0]].permute(2,0,1)
            save_image(feat_rgb,f"renders/segmentation_latest.png")

    # Lossplot
    import matplotlib.pyplot as plt
    plt.plot(losses)
    plt.savefig("figure.png")

    # Save model
    out_dir = os.path.dirname(args.load_checkpoint)
    path = os.path.join(out_dir, 'segmentation', f'iter_{iters}')
    os.makedirs(path, exist_ok=True)
    model.to_ply(path=path, overwrite=True)
    print("Saved model to",path)
