import os
import numpy as np
import tqdm
import argparse

import torch
from torchvision.utils import save_image

from src.model.gaussians import Gaussians
from src.model.m2f import CLASSES,PALETTE,SegmentationModel
from src.data import DataSet
from src.arg import ModelParams,DataParams,TrainParams,PipeLineParams,get_args

if __name__ == '__main__':

    # Get args
    parser = argparse.ArgumentParser("Training 3DGS model on M2F predictions")
    parser.add_argument(
        "--seg-config", type=str, help="Segmentation model config file",
        default="/home/jip/novel-viewpoint-synthesis/submodules/Mask2Former/configs/tdu/semantic-segmentation/jip/maskformer2_swin_large_IN21k_384_bs16_160k_res640.yaml",
    )
    parser.add_argument(
        "--seg-weights", type=str, help="Segmentation model weights",
        default="/home/jip/novel-viewpoint-synthesis/models/model_sem.pth",
    )
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

    print("Reading from 3DU dataset...")
    dataset = DataSet.from_3du(
        device=args.device,
        image_format='seg',
        **vars(data_args)
    )

    # Read classes metadata
    class_names = CLASSES
    class_colors = torch.from_numpy( PALETTE / 255.).to(device=args.device)
    num_classes = len(class_names)

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

    torch.cuda.empty_cache()

    # Only train 'additional_features'
    for group in model.optimizer.param_groups:
        if group['name'] == 'additional_features':
            group['lr'] = 0.0050
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
    pbar = tqdm.trange(args.iterations)
    for it in pbar:

        # Reset grad
        model.optimizer.zero_grad(set_to_none=True)

        # Get next cam to train on
        cam = next(data_cycle)

        # Forward pass
        pkg = model.render(cam, bg=bg)
        feat = pkg.additional_features.view(-1,num_classes)
        loss = loss_fn(feat, cam.gt_image.flatten())

        # Use alpha channel to account for areas with no splats
        loss = loss * pkg.alpha.flatten().detach()
        loss = loss.mean()

        # Backward step
        loss.backward()
        model.optimizer.step()

        # Iter report
        losses.append(loss.item())

        if it%50 == 0 or it == args.iterations-1:
            pkg = model.render(test_cam, bg=bg)
            feat = pkg.additional_features.argmax(dim=-1)

            feat_rgb = class_colors[feat % class_colors.shape[0]].permute(2,0,1)
            save_image(feat_rgb,f"renders/segmentation_latest.png")

    # Save model
    out_dir = os.path.dirname(args.load_checkpoint)
    path = os.path.join(out_dir, 'segmentation', f'iter_{args.iterations}')
    os.makedirs(path, exist_ok=True)
    model.to_ply(path=path, overwrite=True)
    print("Saved model to",path)
