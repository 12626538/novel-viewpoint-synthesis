import os
import numpy as np
import json
import math
import tqdm

from src.model.m2f import SegmentationModel
import cv2
import open3d as o3d
import video3d

def focal2fov(focal:float, pixels:int) -> float:
    """
    From https://github.com/graphdeco-inria/gaussian-splatting/blob/main/utils/graphics_utils.py
    """
    return 2*math.atan(pixels/(2*focal))

def convert_3du(
        video3d_file:str,
        output_dir:str,
        dname_images:str='images',
        dname_segmentations:str='segmentations',
        fname_ply:str='point_cloud.ply',
        fname_json:str='cameras.json',
        start_frame:int=0,
        end_frame:int=-1,
        step_frame:int=1,
        voxel_size:float=0.025,
        segmodel:SegmentationModel=None,
    ):
    """
    Function that createa a .ply point cloud from a video3d file.

    Args:
    - `video3d_file:str` path to the video.v3dc file
    - `outpur_dir:str` root folder to output to, all output will be saved
        from here, and all other arguments are relative to this
    - `dname_images:str` determines the folder name where the images should
        be saved
    - `fname_ply:str` determines the filename where the point cloud should be
        saved. String should end with '.ply'
    - `fname_json:str` determines the filename where the cameras should be
        saved. String should end with `.json`.
    - `start_frame,end_frame,step_frame:int` determines start stop indices and
        step size to iterate dataset. Includes start, excludes stop.
    - `voxel_size:float` determines the downsample voxel size
    - `segmodel:Optional[SegmentationModel]` is the segmentation model to be used

    """

    # Set up output directories
    output_dir = os.path.abspath(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    output_ply = os.path.join(output_dir, fname_ply)
    output_json = os.path.join(output_dir, fname_json)
    output_segmentations = os.path.join(output_dir, dname_segmentations)
    os.makedirs(output_segmentations, exist_ok=True)
    output_images = os.path.join(output_dir, dname_images)
    os.makedirs(output_images, exist_ok=True)

    print("Writing to", output_dir)

    # Set up video reader
    video3d_reader = video3d.Video3DReader(video3d_file, video3d.READ_ALL)

    # Sum pcd of each frame here
    pointcloud = None

    # Output camera parameters here
    f_cameras = open(output_json, "w")

    # Keep track of camera positions and orientations
    camera_locs = []
    camera_lookats = []
    lookat = np.array([0,0,1])

    pbar = tqdm.tqdm(desc="Converting 3DU data", mininterval=1, unit="frame")

    segvideo = None
    segmentations = []

    # Start iterating video reader
    uid = 0
    for frame_idx, frame in enumerate(video3d_reader):

        if frame is None: break;
        if frame_idx<start_frame or frame_idx%step_frame!=0: continue;
        if end_frame > 0 and frame_idx>=end_frame: break;

        pbar.update(1)

        # Read data
        info = frame.info()
        img = frame.img().astype(np.uint8)
        depth = np.squeeze(frame.depth())
        confidence = np.squeeze(frame.confidence())

        if 0 in img.shape:
            continue

        uid += 1

        # Get parameters
        intrinsics = np.squeeze(np.array(info['cameraIntrinsics'])).T
        extrinsics = np.squeeze(np.array(info['localToWorld'])).T
        extrinsics_inv = np.linalg.inv(extrinsics) # world-to-view

        # Set camera
        H,W = img.shape[:2]
        # Camera intrinsics are based on the original image, which is
        # twice the resolution of `img`!
        H*=2
        W*=2

        R = extrinsics_inv[:3,:3]
        t = extrinsics_inv[:3,3]

        fname = f'frame{frame_idx:06d}.png'
        # Save frame
        img = frame.img().astype(np.uint8)
        if not os.path.isfile(os.path.join(output_images, fname)):
            cv2.imwrite(os.path.join(output_images, fname), img)

        if segmodel is not None:
            labels,colors,blend = segmodel.segment(img)

            if segvideo is None:
                segvideo = cv2.VideoWriter(
                    os.path.join(output_dir,'segmentation.mp4'),
                    fourcc=cv2.VideoWriter_fourcc(*'mp4v'),
                    fps=round(30/step_frame) if step_frame < 30 else 1,
                    isColor=True, frameSize=(img.shape[1],img.shape[0]),
                )
            segvideo.write(blend)

            cv2.imwrite( os.path.join(output_segmentations, fname), labels )

        f_cameras.write(json.dumps({
            "fname": fname,
            "fovx": focal2fov(intrinsics[0,0], W),
            "fovy": focal2fov(intrinsics[1,1], H),
            "cx_frac": intrinsics[0,2] / W,
            "cy_frac": intrinsics[1,2] / H,
            "R": R.tolist(),
            "t": t.tolist(),
        })+"\n")

        camera_locs.append( - R.T @ t)
        camera_lookats.append( R.T @ lookat )

        if 0 in depth.shape:
            continue

        # Reszie image to match depth info
        row_ratio = depth.shape[0] / img.shape[0]
        col_ratio = depth.shape[1] / img.shape[1]
        img = cv2.resize(img, (depth.shape[1], depth.shape[0]))

        # Convert to open3d Image instance
        color_image = o3d.geometry.Image(np.array(img[:,:,::-1]))

        # Get depth image based on confidence
        depth_raw = depth.astype(np.uint16)
        depth_image = np.zeros((depth.shape[0], depth.shape[1]), dtype=np.uint16)
        depth_image[confidence >= 2] = depth_raw[confidence >= 2]
        depth_image = o3d.geometry.Image(depth_image)

        # Merge color and depth into RGBD
        rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
            color_image, depth_image, depth_scale=1000, depth_trunc=5.0,convert_rgb_to_intensity=False)

        # Update intrinsics after rescaling
        fx = intrinsics[0, 0] * col_ratio * 0.5
        fy = intrinsics[1, 1] * row_ratio * 0.5
        cx = intrinsics[0, 2] * col_ratio * 0.5
        cy = intrinsics[1, 2] * row_ratio * 0.5
        width = depth.shape[1]
        height = depth.shape[0]

        # Get pointcloud in 3D camera coordinates
        pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
            rgbd_image,
            o3d.camera.PinholeCameraIntrinsic(width, height, fx, fy, cx, cy),
        )

        # Convert points to world coordinates
        pcd.transform(extrinsics)

        # Add pointcloud to running reconstruction
        if pointcloud is None:
            pointcloud = pcd
        else:
            pointcloud += pcd

    pbar.close()
    if segvideo is not None:
        segvideo.release()


    print("Number of frames/cameras:",uid)

    f_cameras.close()

    # Create 'pointcloud' from camear locations and lookats
    camera_locs = np.stack(camera_locs, axis=0)
    camera_lookats = np.stack(camera_lookats, axis=0)
    camera_pcd = o3d.geometry.PointCloud(points=o3d.utility.Vector3dVector(camera_locs))
    camera_pcd.normals = o3d.utility.Vector3dVector(camera_lookats)
    fname = os.path.join( os.path.dirname(output_ply), 'cameras.ply' )
    o3d.io.write_point_cloud(fname, camera_pcd, write_ascii=True)

    # Reduce to more useful construction
    pointcloud = pointcloud.voxel_down_sample(voxel_size=voxel_size)
    o3d.io.write_point_cloud(output_ply, pointcloud, write_ascii=True)
    print("Final point cloud size:", len(pointcloud.points))


if __name__ == '__main__':
    import sys
    import argparse

    parser = argparse.ArgumentParser("3DU preprocessing")
    parser.add_argument("video3d_file",type=str,help="Input file, should be .v3dc file")
    parser.add_argument("output_dir",type=str,help="Output directory")

    parser.add_argument("-s", "--start-frame", type=int,default=0, help="Start frame")
    parser.add_argument("-e", "--end-frame", type=int,default=-1, help="End frame, default is last")
    parser.add_argument("-k", "--step-frame", type=int, default=1, help="Use every k-th frame, default is every")

    parser.add_argument("-v", "--voxel-size", type=float, default=0.025, help="Voxel downsample size")

    parser.add_argument("--do-segmentation", action='store_true', default=False)
    parser.add_argument(
        "--seg-config", type=str, help="Segmentation model config file",
        default="/home/jip/novel-viewpoint-synthesis/submodules/Mask2Former/configs/tdu/semantic-segmentation/minh/maskformer2_swin_large_IN21k_384_bs16_160k_res640.yaml",
    )
    parser.add_argument(
        "--seg-weights", type=str, help="Segmentation model weights",
        default="/home/jip/novel-viewpoint-synthesis/models/model_sem.pth",
    )

    args = parser.parse_args()

    segmodel = None
    if args.do_segmentation:
        segmodel = SegmentationModel(
            config_file=args.seg_config,
            model_weights=args.seg_weights
        )

    convert_3du(
        video3d_file=args.video3d_file,
        output_dir=args.output_dir,
        start_frame=args.start_frame,
        end_frame=args.end_frame,
        step_frame=args.step_frame,
        voxel_size=args.voxel_size,
        segmodel=segmodel,
    )
