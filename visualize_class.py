import argparse
import os
import numpy as np

import torch
from torchvision.utils import save_image

from src.model.gaussians import Gaussians,RenderPackage
from src.model.m2f import CLASSES,PALETTE
from src.data import DataSet,get_rotating_dataset
from src.arg import ModelParams,DataParams,PipeLineParams,get_args
from src.camera import Camera
from src.utils.smoothing_utils import smooth_camera_path
from src.data import DataSet

import cv2

def get_dataset(args,data_args):
    # 3DU datasets are from intrinsics and extrinsics
    if "3du" in data_args.source_path:
        dataset = DataSet.from_3du(device=args.device, **vars(data_args))

    # Paper datasets are COLMAP formatted
    else:
        dataset = DataSet.from_colmap(device=args.device, **vars(data_args))

    return dataset

if __name__ == '__main__':

    parser = argparse.ArgumentParser("Training Novel Viewpoint Synthesis")

    parser.add_argument(
        "--out-dir", type=str,
        default=None,
        help="Where to output images to, default is `renders` folder inside checkpoint directory"
    )
    parser.add_argument(
        '--background',
        default="black",
        choices=['black','white', 'random'],
        help='Background of rendered images'
    )
    parser.add_argument(
        '--fps', type=int,
        default=None,
        help='when generating videos, use this framerate'
    )
    parser.add_argument(
        '--dataset-fps', type=int,
        default=15,
        help='Framerate used by dataset, used for smoothing'
    )
    parser.add_argument(
        '--T', type=float,
        default=40,
        help='When generating video, make it of this length'
    )
    parser.add_argument(
        '--glob-scale', type=float,
        default=1,
        help="Global scaling factor for the gaussians, used for rendering"
    )
    parser.add_argument(
        '--center-point', type=float, nargs=3,
        default=[0,0,0],
        help="For 'rotation' method, rotate around this point"
    )
    parser.add_argument(
        '--rotation-axis', type=float, nargs=3,
        default=[0,1,0],
        help="For 'rotation' method, rotate around this axis"
    )
    parser.add_argument(
        '--rotation-distance', type=float,
        default=5,
        help="For 'rotation' method, rotate this distance from center point"
    )

    args,(data_args,model_args,pipeline_args) = get_args(
        DataParams,ModelParams,PipeLineParams,
        parser=parser
    )

    try:
        torch.cuda.set_device(args.device)
    except:
        print(f"!Warning! Could not use device {args.device}, falling back to cuda:0.")
        args.device = 'cuda:0'

    # Get background
    if args.background == "black":
        args.background = torch.zeros(3, device=args.device)
    elif args.background == "white":
        args.background = torch.ones(3, device=args.device)
    else:
        args.background = torch.rand(3, device=args.device)

    # Use checkpoint
    if not os.path.exists(pipeline_args.load_checkpoint):
        print(f"Checkpoint {pipeline_args.load_checkpoint} not found")
        exit(1)

    # Save results here
    if args.out_dir is None:
        args.out_dir = os.path.join(
            os.path.dirname(pipeline_args.load_checkpoint),
            'renders'
        )
        os.makedirs(args.out_dir, exist_ok=True)
        print("Outputting evaluations at", args.out_dir)

    args.fps = args.dataset_fps if args.fps is None else args.fps

    # Get dataset
    # 3DU datasets have source path defined
    if "3du" in args.source_path:
        dataset = DataSet.from_3du(
            device=args.device,
            **vars(data_args)
        )

        # Read classes metadata
        class_names = CLASSES
        class_colors = PALETTE / 255.
        class_colors = torch.tensor(class_colors, device=args.device, dtype=torch.float32)


        poses = smooth_camera_path(dataset.cameras, num_poses=len(dataset)*args.dataset_fps)

        dataset = DataSet([
            Camera(
                R=pose[:3,:3],
                t=pose[:3,3],
                znear=1e-3,
                fovx=np.deg2rad(66),
                fovy=np.deg2rad(37),
                H=720, W=1280,
            )
            for pose in poses
        ])

    # Other dataset pans around center point
    else:
        rotax = np.array(args.rotation_axis)
        dataset = get_rotating_dataset(
            distance_from_center=args.rotation_distance,
            center_point=np.array(args.center_point),
            rotation_axis=rotax / np.linalg.norm(rotax),
            num_cameras=round(args.T * args.fps),
            H=720,W=1280
        )
        class_colors = torch.tensors([ [1,0,0], [0,1,0] ], device=args.device, dtype=torch.float32)

    # Load model
    print(f"Loading .ply model from {pipeline_args.load_checkpoint}")
    model = Gaussians.from_ply(
        device=args.device,
        path_to_ply_file=pipeline_args.load_checkpoint,
        **vars(model_args),
    )
    # Use full SH degree
    model.sh_degree_current = model.sh_degree_max

    num_additional_features = model.additional_features.shape[-1]
    background = torch.zeros(3+num_additional_features,device=args.device)

    try:
        unk_class = class_names.index('unknown')
    except ValueError:
        unk_class = 0
    background[3+unk_class] = 1.

    print("Rendering video...")
    H,W = 720,1280
    alpha_segementation = .4

    # Set up output video writers
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_rgb = cv2.VideoWriter(
        filename=os.path.join(args.out_dir, 'video.mp4'),
        fourcc=fourcc, isColor=True,
        fps=args.fps, frameSize=(W,H),
    )
    out_segmentation = cv2.VideoWriter(
        filename=os.path.join(args.out_dir, 'segmentation.mp4'),
        fourcc=fourcc, isColor=True,
        fps=args.fps, frameSize=(W,H),
    )
    out_depth = cv2.VideoWriter(
        filename=os.path.join(args.out_dir, 'depth.mp4'),
        fourcc=fourcc, isColor=False,
        fps=args.fps, frameSize=(W,H),
    )

    n = len(dataset)
    for i,camera in enumerate(dataset):
        print(f"\rRendering... {i+1: 6d}/{n: 6d}",end="",flush=True)

        # Render camera
        pkg = model.render(
            camera,
            bg=background,
            glob_scale=args.glob_scale,
            return_depth=True
        )

        # RGB
        rgb = pkg.rendering.permute(1,2,0)
        rgb_np = rgb.detach().cpu().numpy()

        # FEATURES
        feat = pkg.additional_features.argmax(dim=-1) % class_colors.shape[0]
        feat_rgb = class_colors[ feat ]
        feat_rgb_np = feat_rgb.detach().cpu().numpy()

        # DEPTH
        depth_np = pkg.depth.detach().cpu().numpy()
        depth_np = depth_np / depth_np.max()

        # Save results
        out_rgb.write( cv2.cvtColor( (rgb_np*255).astype(np.uint8), cv2.COLOR_RGB2BGR) )

        segmentation = feat_rgb_np * alpha_segementation + rgb_np * (1-alpha_segementation)
        out_segmentation.write( cv2.cvtColor( (segmentation *255).astype(np.uint8), cv2.COLOR_RGB2BGR) )

        out_depth.write( (depth_np *255).astype(np.uint8) )

    print("\nDone.")
