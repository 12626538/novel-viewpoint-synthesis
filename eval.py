import argparse
import os
import numpy as np

import torch
from torchvision.utils import save_image
from torcheval.metrics import PeakSignalNoiseRatio

from src.model.gaussians import Gaussians,RenderPackage
from src.data import DataSet
from src.utils.loss_utils import L1Loss,DSSIMLoss,CombinedLoss
from src.arg import ModelParams,DataParams,TrainParams,PipeLineParams

if __name__ == '__main__':

    parser = argparse.ArgumentParser("Training Novel Viewpoint Synthesis")
    dp = DataParams(parser)
    mp = ModelParams(parser)
    pp = PipeLineParams(parser)

    parser.add_argument(
        '--video', action='store_true',
        default=False,
        help='Interpret dataset as a video with provided framerate'
    )
    parser.add_argument(
        '--fps', type=int,
        default=30,
        help='If --video flag is set, generate video with this framerate'
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

    print(f"Loading .ply model from {pipeline_args.load_checkpoint}")
    model = Gaussians.from_ply(
        device=args.device,
        path_to_ply_file=pipeline_args.load_checkpoint,
        **vars(model_args)
    )

    print(model.colors.shape)
    1/0

    # 3DU datasets are from intrinsics and extrinsics
    if data_args.source_path[:-1].endswith("3du_data_"):
        dataset = DataSet.from_intr_extr(device=args.device, **vars(data_args))

    # Paper datasets are COLMAP formatted
    else:
        dataset = DataSet.from_colmap(device=args.device, **vars(data_args))

    if args.video:

        print("Rendering video...")

        os.environ["IMAGEIO_FFMPEG_EXE"] = "/usr/bin/ffmpeg"
        from moviepy.editor import VideoClip

        cameras = sorted(dataset.cameras, key=lambda cam: cam.name)
        T = len(cameras)/args.fps
        H = max(cam.H for cam in cameras)+5
        W = max(cam.W for cam in cameras)+5

        def make_frame(t):
            out = np.zeros((H,2*W,3),dtype=np.uint8)

            # Get camera at current timestep
            i = round((t / T)*(len(cameras)-1))
            camera = cameras[i]

            A = (camera.gt_image.detach().permute(1,2,0).cpu().numpy() * 255).astype(np.uint8)

            # Render camera
            render = model.render(camera, bg=torch.zeros(3,device=args.device)).rendering.permute(1,2,0)
            B = (render.detach().cpu().numpy()*255).astype(np.uint8)

            out[:A.shape[0], :A.shape[1]] = A
            out[:B.shape[0], W:W+B.shape[1]] = B
            return out

        clip = VideoClip(make_frame, duration=T)
        clip.write_videofile("render.mp4",fps=args.fps)
