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
    Render a list of cameras using model and save in out_dir

    Creates folder if it does not exist yet.
    """
    os.makedirs(out_dir, exist_ok=True)

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
        choices=['images','video', 'blendvideo', 'smoothvideo', 'rotation'],
        help='How to render images, default is all views as seperate images'
    )
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
        dataset = get_dataset(args, data_args)

        print("Rendering images...")

        for part in ('train', 'test'):
            render_images(
                cameras=dataset.iter(part),
                model=model,
                out_dir=os.path.join(args.out_dir,part),
                glob_scale=args.glob_scale,
                background=args.background
            )
    # Render images as video
    elif args.method == "video":
        dataset = get_dataset(args, data_args)

        cameras = sorted(dataset.cameras, key=lambda cam: cam.name)
        T = len(cameras)/args.dataset_fps
        H = min(cam.H for cam in cameras)
        W = min(cam.W for cam in cameras)

        def render_frame(t):
            # Get camera at current timestep
            i = round((t / T)*(len(cameras)-1))
            camera = cameras[i]

            # Ground truth image
            if camera.gt_image is not None:
                A = (camera.gt_image.detach().permute(1,2,0).cpu().numpy() * 255).astype(np.uint8)[:H,:W]
            else:
                A = np.zeros((H,W,3))

            # Render camera
            pkg = model.render(camera, bg=torch.zeros(3,device=args.device), glob_scale=args.glob_scale)
            render = pkg.rendering.permute(1,2,0)
            B = (render.detach().cpu().numpy()*255).astype(np.uint8)[:H,:W]

            # Stack horizontally
            out = np.hstack((A,B), dtype=np.uint8)

            return out

        print("Rendering video...")
        render_video(
            render_frame=render_frame,
            fname=os.path.join(args.out_dir, "reconstruction.mp4"),
            duration=T,
            fps=args.dataset_fps if args.fps is None else args.fps
        )

    # Render images as video, layer them using the alpha value of the rendering
    elif args.method == "blendvideo":
        dataset = get_dataset(args, data_args)

        cameras = sorted(dataset.cameras, key=lambda cam: cam.name)
        T = len(cameras)/args.dataset_fps
        H = min(cam.H for cam in cameras)
        W = min(cam.W for cam in cameras)

        def render_frame(t):
            # Get camera at current timestep
            i = round((t / T)*(len(cameras)-1))
            camera = cameras[i]

            # Ground truth image
            if camera.gt_image is not None:
                A = (camera.gt_image.detach().permute(1,2,0).cpu().numpy() * 255).astype(np.uint8)[:H,:W]
            else:
                A = np.zeros((H,W,3))

            # Render camera
            pkg = model.render(camera, bg=torch.zeros(3,device=args.device), glob_scale=args.glob_scale)
            render = pkg.rendering.permute(1,2,0)
            B = (render.detach().cpu().numpy()*255).astype(np.uint8)[:H,:W]

            alpha = pkg.alpha.detach().cpu().numpy().reshape(H,W,1)

            out = alpha* B + (1-alpha)* A

            return out

        print("Rendering video...")
        render_video(
            render_frame=render_frame,
            fname=os.path.join(args.out_dir, "reconstruction_blended.mp4"),
            duration=T,
            fps=args.dataset_fps if args.fps is None else args.fps
        )

    # Render images as video
    elif args.method == "smoothvideo":

        from src.utils.smoothing_utils import smooth_camera_path

        dataset = get_dataset(args, data_args)

        print("Smoothing camera path...")
        fps = args.dataset_fps if args.fps is None else args.fps
        num_frames = args.T * fps

        poses = smooth_camera_path(dataset.cameras, num_poses=num_frames)
        num_poses = len(poses)

        def render_frame(t):
            # Get pose at current timestep
            i = round((t / args.T)*(num_poses-1))
            pos = poses[i]

            camera = Camera(
                R=pos[:3,:3],
                t=pos[:3,3],
                znear=1e-3,
                fovx=np.deg2rad(66),
                fovy=np.deg2rad(37),
                H=1080, W=1920
            )

            render = model.render(camera, bg=torch.zeros(3,device=args.device), glob_scale=args.glob_scale).rendering.permute(1,2,0)
            out = (render.detach().cpu().numpy() * 255).astype(np.uint8)

            return out

        print("Rendering video...")
        render_video(
            render_frame=render_frame,
            fname=os.path.join(args.out_dir, "reconstruction_smooth.mp4"),
            duration=args.T,
            fps=fps
        )

    # Render rotating view
    elif args.method == "rotation":
        rotax = np.array(args.rotation_axis)
        dataset = get_rotating_dataset(
            distance_from_center=args.rotation_distance,
            center_point=np.array(args.center_point),
            rotation_axis=rotax / np.linalg.norm(rotax),
            num_cameras=round(args.T * args.fps),
        )


        cameras = sorted(dataset.cameras, key=lambda cam: cam.name)
        H = min(cam.H for cam in cameras)
        W = min(cam.W for cam in cameras)

        def render_frame(t):
            out = np.zeros((H,W,3),dtype=np.uint8)

            # Get camera at current timestep
            i = round((t / args.T)*(len(cameras)-1))
            camera = cameras[i]

            # Render camera
            render = model.render(camera, bg=torch.zeros(3,device=args.device), glob_scale=args.glob_scale).rendering.permute(1,2,0)
            A = (render.detach().cpu().numpy()*255).astype(np.uint8)[:H,:W]

            out[:H, :W] = A
            return out

        print("Rendering video...")
        render_video(
            render_frame=render_frame,
            fname=os.path.join(args.out_dir, "rotation.mp4"),
            duration=args.T,
            fps=args.fps
        )
