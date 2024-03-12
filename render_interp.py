import argparse
import os
import numpy as np
import math

import torch
from torchvision.utils import save_image

from src.model.gaussians import Gaussians,RenderPackage
from src.data import DataSet
from src.camera import Camera
from src.arg import ModelParams,DataParams,PipeLineParams,get_args

from scipy.spatial.transform import Rotation

def viewmat(tpl):
    R,loc = tpl
    pose = np.eye(4)
    pose[:3,:3] = R
    pose[:3,3] = -loc@R.T
    return pose

def rotmat(lookat, up=np.array([0,-1,0])):
    # Z-axis is lookat
    vec2 = lookat / np.linalg.norm(lookat)

    # X-axis is perp. to Z and up axis
    vec0 = np.cross(up, vec2)
    vec0 /= np.linalg.norm(vec0)

    # Y axis is perp to X,Z axes
    vec1 = np.cross(vec2, vec0)
    vec1 /= np.linalg.norm(vec1)

    return np.stack([vec0,vec1,vec2],1).T



def catmull_rom_spline(P0,P1,P2,P3, num_points, alpha) -> np.ndarray:
    """
    FROM https://en.wikipedia.org/wiki/Centripetal_Catmull%E2%80%93Rom_spline
    Compute the points in the spline segment
    :param P0, P1, P2, and P3: The (x,y) point pairs that define the Catmull-Rom spline
    :param num_points: The number of points to include in the resulting curve segment
    :param alpha: 0.5 for the centripetal spline, 0.0 for the uniform spline, 1.0 for the chordal spline.
    :return: The points
    """

    # Calculate t0 to t4. Then only calculate points between P1 and P2.
    # Reshape linspace so that we can multiply by the points P0 to P3
    # and get a point for each value of t.
    def tj(ti: float, pi: tuple, pj: tuple) -> float:
        xi, yi, zi = pi
        xj, yj, zj = pj
        dx, dy, dz = xj - xi, yj - yi, zj - zi
        l = (dx ** 2 + dy ** 2 + dz ** 2) ** 0.5
        return ti + l ** alpha

    t0: float = 0.0
    t1: float = tj(t0, P0, P1)
    t2: float = tj(t1, P1, P2)
    t3: float = tj(t2, P2, P3)
    t = np.linspace(t1, t2, num_points).reshape(num_points, 1)

    A1 = (t1 - t) / (t1 - t0) * P0 + (t - t0) / (t1 - t0) * P1
    A2 = (t2 - t) / (t2 - t1) * P1 + (t - t1) / (t2 - t1) * P2
    A3 = (t3 - t) / (t3 - t2) * P2 + (t - t2) / (t3 - t2) * P3
    B1 = (t2 - t) / (t2 - t0) * A1 + (t - t0) / (t2 - t0) * A2
    B2 = (t3 - t) / (t3 - t1) * A2 + (t - t1) / (t3 - t1) * A3
    points = (t2 - t) / (t2 - t1) * B1 + (t - t1) / (t2 - t1) * B2
    return points

def smooth_directions(directions, points, alpha):
    num_points = round( points / (len(directions)-3) )
    return np.concatenate(
        [ catmull_rom_spline(*directions[i:i+4], num_points=num_points, alpha=alpha)
         for i in range(len(directions)-3)
        ]
    )

def smooth_camera_path(cameras, num_poses=600, alpha:float=1) -> list[np.ndarray]:
    """
    Interpolate between cameras to create a smooth path

    Args:
    - `cameras:list[Camera]` A sorted list of camera instances
    - `num_poses:int` Total number of poses to interpolate
    - `alpha:float` Smoothing factor for Catmull Rom spline function

    Returns:
    - `poses:np.ndarray` A list of np arrays where the ith entry is the 4x4
        interpolated world-to-view matrix
    """
    # Use 3D 'lookAt' vector for direction
    lookat = np.array([0,0,1])

    # get lookAt for every camera
    lookats = np.array([cam.R.T @ lookat for cam in cameras])

    lookats[:,1] /= 2
    lookats /= np.linalg.norm(lookats, axis=-1, keepdims=True)

    n_combine = 10

    # Group take the mean of every N cameras, extrapolate additional start and end value
    lookats = lookats[:(lookats.shape[0]//n_combine)*n_combine].reshape(-1,n_combine,3).mean(axis=1)
    lookats = np.pad(lookats, ((1,1),(0,0)),'reflect',reflect_type='odd')

    # Do the same for camera locations
    locs = np.array([cam.loc for cam in cameras])
    locs = locs[:(locs.shape[0]//n_combine)*n_combine].reshape(-1,n_combine,3).mean(axis=1)
    locs = np.pad(locs, ((1,1),(0,0)),'reflect',reflect_type='odd')

    # Smooth everything out
    smooth_lookats = smooth_directions(lookats, num_poses, alpha)
    Rs = map(rotmat, smooth_lookats)
    smooth_locs = smooth_directions(locs, num_poses, alpha)

    # Reconstruct matrices
    poses = map(viewmat, zip(Rs,smooth_locs))

    return list(poses)



os.environ["IMAGEIO_FFMPEG_EXE"] = "/usr/bin/ffmpeg"
from moviepy.editor import VideoClip

if __name__ == '__main__':

    parser = argparse.ArgumentParser("Training Novel Viewpoint Synthesis")

    parser.add_argument(
        '--fps', type=int,
        default=30,
        help='Framerate of output video'
    )
    parser.add_argument(
        '--glob-scale', type=float,
        default=1,
        help="Global scaling factor for the gaussians, used for rendering"
    )
    args, (data_args, model_args, pipeline_args) = get_args(DataParams, ModelParams, PipeLineParams, parser=parser)

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

    # Get dataset
    if "3du" in data_args.source_path:
        dataset = DataSet.from_3du(device=args.device, **vars(data_args))
    else:
        dataset = DataSet.from_colmap(device=args.device, **vars(data_args))

    cameras = sorted(dataset.cameras, key=lambda cam: cam.name)

    # Original data is 15 fps, but only every second frame is taken
    T = len(dataset) / 15 * 2
    num_frames = T*args.fps

    poses = smooth_camera_path(cameras, num_poses=num_frames)
    num_poses = len(poses)


    def make_frame(t):

        # Get pose at current timestep
        i = round((t / T)*(num_poses-1))
        pos = poses[i]

        camera = Camera(
            R=pos[:3,:3],
            t=pos[:3,3],
            znear=1e-3,
            fovx=np.deg2rad(66),
            fovy=np.deg2rad(33),
            H=1080, W=1920
        )

        render = model.render(camera, bg=torch.zeros(3,device=args.device), glob_scale=args.glob_scale).rendering.permute(1,2,0)
        A = (render.detach().cpu().numpy() * 255).astype(np.uint8)

        return A

    print(f"Rendering video ({round(T*args.fps)} frames)...")
    clip = VideoClip(make_frame, duration=T)
    clip.write_videofile(
        os.path.join(out_dir, "smoothed_reconstruction.mp4"),
        fps=args.fps,
        threads=5,
        logger=None,
    )
    print("Written video to", os.path.join(out_dir, "smoothed_reconstruction.mp4"))
