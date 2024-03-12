import argparse
import os
import numpy as np

import torch
from torchvision.utils import save_image

from src.model.gaussians import Gaussians,RenderPackage
from src.data import DataSet,get_rotating_dataset
from src.arg import ModelParams,DataParams,PipeLineParams,get_args
from src.camera import Camera

os.environ["IMAGEIO_FFMPEG_EXE"] = "/usr/bin/ffmpeg"
from moviepy.editor import VideoClip

def get_dataset(args,data_args):
    # 3DU datasets are from intrinsics and extrinsics
    if "3du" in data_args.source_path:
        dataset = DataSet.from_3du(device=args.device, **vars(data_args))

    # Paper datasets are COLMAP formatted
    else:
        dataset = DataSet.from_colmap(device=args.device, **vars(data_args))

    return dataset

def render_images(
        cameras:list[Camera],
        model:Gaussians,
        out_dir:str,
        glob_scale:float,
        background:torch.Tensor
    ):
    """
    Render a list of cameras using model and save in out_dir.
    """

    for camera in cameras:
        render_pkg = model.render(camera, glob_scale=glob_scale, bg=background)
        save_image(render_pkg.rendering, os.path.join(out_dir,camera.name))


def render_video(
        render_frame,
        fname:str,
        duration:float,
        fps:int
    ):
        clip = VideoClip(render_frame, duration=duration)
        clip.write_videofile(
            fname,
            fps=fps,
            threads=5,
            logger=None,
        )
        print("Written video to", fname)

if __name__ == '__main__':

    parser = argparse.ArgumentParser("Training Novel Viewpoint Synthesis")

    parser.add_argument(
        'method',
        default="images",
        choices=['images','video', 'smoothvideo', 'alphablend', 'rotation'],
        help='How to render images, default is all views as seperate images'
    )
    parser.add_argument(
        '--fps', type=int,
        default=48,
        help='when generating videos, use this framerate'
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
        '--background',
        default="black",
        choices=['black','white', 'random'],
        help='Background of rendered images'
    )
    parser.add_argument(
        "--out-dir", type="str",
        default=None,
        help="Where to output images to, default is `renders` folder inside checkpoint directory"
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

    # Load model
    print(f"Loading .ply model from {pipeline_args.load_checkpoint}")
    model = Gaussians.from_ply(
        device=args.device,
        path_to_ply_file=pipeline_args.load_checkpoint,
        **vars(model_args)
    )
    # Use full SH degree
    model.sh_degree_current = model.sh_degree_max

    # Simple render images and save to out_dir
    if args.method == "images":
        dataset = get_dataset(args,data_args)
        render_images(
            cameras=dataset.cameras,
            model=model,
            out_dir=args.out_dir,
            glob_scale=args.glob_scale,
            background=args.background
        )

    # Render images as video
    elif args.method == "video":

    exit()

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

            if args.alpha_blend:
                # ALPHA BLENDING
                alpha = pkg.alpha.detach().cpu().numpy()[:H,:W,None]
                out = np.zeros((H,W,3),dtype=np.uint8)
                out[:H, :W] = alpha* B + (1-alpha)* A
            else:
                # SIDE-BY-SIDE
                out = np.zeros((H,W,3),dtype=np.uint8)

                out[:H, :W] = B[:,:W//2]
                out[:H, -W//2:] = B[:,-W//2:]

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
            distance_from_center=5,
            center_point=np.array([0,0,0]),
            rotation_axis=np.array([0,1,1])/np.sqrt(2),
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
        clip.write_videofile(
            os.path.join(out_dir, "render.mp4"),
            fps=args.fps,
            threads=5,
            logger=None,
        )
        print("Written video to", os.path.join(out_dir, "render.mp4"))
