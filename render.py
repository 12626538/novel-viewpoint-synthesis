import argparse
import os
import numpy as np

import torch
from torchvision.utils import save_image

from src.model.gaussians import Gaussians,RenderPackage
from src.data import DataSet,get_rotating_dataset
from src.arg import ModelParams,DataParams,PipeLineParams

os.environ["IMAGEIO_FFMPEG_EXE"] = "/usr/bin/ffmpeg"
from moviepy.editor import VideoClip

if __name__ == '__main__':

    parser = argparse.ArgumentParser("Training Novel Viewpoint Synthesis")
    dp = DataParams(parser)
    mp = ModelParams(parser)
    pp = PipeLineParams(parser)

    parser.add_argument(
        '--reconstruct-video', action='store_true',
        default=False,
        help='Interpret dataset as a video with provided framerate'
    )
    parser.add_argument(
        '--fps', type=int,
        default=30,
        help='when generating videos, use this framerate'
    )
    parser.add_argument(
        '--glob-scale', type=float,
        default=1,
        help="Global scaling factor for the gaussians, used for rendering"
    )

    args = parser.parse_args()

    data_args = dp.extract(args)
    model_args = mp.extract(args)
    pipeline_args = pp.extract(args)

    try:
        torch.cuda.set_device(args.device)
    except:
        print(f"!Warning! Could not use device {args.device}, falling back to cuda:0.")
        args.device = 'cuda:0'

    # Use checkpoint
    if not os.path.exists(pipeline_args.load_checkpoint):
        print(f"Checkpoint {pipeline_args.load_checkpoint} not found")
        exit(1)

    # Save results here
    out_dir = os.path.join(
        os.path.dirname(pipeline_args.load_checkpoint),
        'renders'
    )
    os.makedirs(out_dir, exist_ok=True)
    print("Outputting evaluations at", out_dir)

    print(f"Loading .ply model from {pipeline_args.load_checkpoint}")
    model = Gaussians.from_ply(
        device=args.device,
        path_to_ply_file=pipeline_args.load_checkpoint,
        **vars(model_args)
    )

    model.sh_degree_current = model.sh_degree_max

    if args.reconstruct_video:

        # 3DU datasets are from intrinsics and extrinsics
        if "3du" in data_args.source_path:
            dataset = DataSet.from_3du(device=args.device, **vars(data_args))

        # Paper datasets are COLMAP formatted
        else:
            dataset = DataSet.from_colmap(device=args.device, **vars(data_args))

        print("Rendering video...")

        cameras = sorted(dataset.cameras, key=lambda cam: cam.name)
        T = len(cameras)/args.fps
        H = min(cam.H for cam in cameras)
        W = min(cam.W for cam in cameras)

        def make_frame(t):

            # Get camera at current timestep
            i = round((t / T)*(len(cameras)-1))
            camera = cameras[i]

            A = (camera.gt_image.detach().permute(1,2,0).cpu().numpy() * 255).astype(np.uint8)[:H,:W]

            # Render camera
            pkg = model.render(camera, bg=torch.zeros(3,device=args.device), glob_scale=args.glob_scale)
            render = pkg.rendering.permute(1,2,0)
            B = (render.detach().cpu().numpy()*255).astype(np.uint8)[:H,:W]

            # SIDE-BY-SIDE
            out = np.zeros((H,2*W,3),dtype=np.uint8)
            out[:H, :W] = A
            out[:H, -W:] = B


            # ALPHA BLENDING
            alpha = pkg.alpha.detach().cpu().numpy()[:H,:W,None]
            out = np.zeros((H,W,3),dtype=np.uint8)
            out[:H, :W] = alpha* B + (1-alpha)* A
            return out

        clip = VideoClip(make_frame, duration=T)
        clip.write_videofile(
            os.path.join(out_dir, "reconstruction.mp4"),
            fps=args.fps,
            threads=5,
            logger=None,
        )
        print("Written video to", os.path.join(out_dir, "reconstruction.mp4"))

    else:
        dataset = get_rotating_dataset(
            distance_from_center=0,#5,
            center_point=np.array([0,1,0]),
            num_cameras=240
        )

        print("Rendering video...")


        cameras = sorted(dataset.cameras, key=lambda cam: cam.name)
        T = len(cameras)/args.fps
        H = min(cam.H for cam in cameras)
        W = min(cam.W for cam in cameras)

        def make_frame(t):
            out = np.zeros((H,W,3),dtype=np.uint8)

            # Get camera at current timestep
            i = round((t / T)*(len(cameras)-1))
            camera = cameras[i]

            # Render camera
            render = model.render(camera, bg=torch.zeros(3,device=args.device), glob_scale=args.glob_scale).rendering.permute(1,2,0)
            A = (render.detach().cpu().numpy()*255).astype(np.uint8)[:H,:W]

            out[:H, :W] = A
            return out

        clip = VideoClip(make_frame, duration=T)
        clip.write_videofile(os.path.join(out_dir, "render.mp4"),fps=args.fps)
