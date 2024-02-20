import os
import numpy as np
import open3d as o3d
import video3d
import cv2
import json
import math

import tqdm

def focal2fov(focal:float, pixels:int) -> float:
    """
    From https://github.com/graphdeco-inria/gaussian-splatting/blob/main/utils/graphics_utils.py
    """
    return 2*math.atan(pixels/(2*focal))

def convert_3du(
        video3d_file:str,
        output_dir:str,
        dname_images:str='images',
        fname_ply:str='pointcloud.ply',
        fname_json:str='cameras.json',
    ):
    """
    Function that createa a .ply point cloud from a video3d file.

    :video3d_file: (str) path to the video.v3dc file
    :outpur_dir: (str) root folder to output to, all output will be saved
        from here, and all other arguments are relative to this
    :dname_images: (str) determines the folder name where the images should
        be saved
    :fname_ply: (str) determines the filename where the point cloud should be
        saved. String should end with '.ply'
    :fname_intr: (str) determines the filename where the intrinsics should be
        saved. String should end with '.npy'
    :fname_extr: (str) determines the filename where the extrinsics should be
        saved. String should end with '.npy'

    """

    # Set up output directories
    output_dir = os.path.abspath(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    output_ply = os.path.join(output_dir, fname_ply)
    output_json = os.path.join(output_dir, fname_json)
    output_images = os.path.join(output_dir, dname_images)
    os.makedirs(output_images, exist_ok=True)

    print("Writing to", output_dir)

    # Set up video reader
    video3d_reader = video3d.Video3DReader(video3d_file, video3d.READ_ALL)

    # Set up output variables
    pointcloud = None

    f_cameras = open(output_json, "w")

    frame_id = 0
    for i, frame in tqdm.tqdm(enumerate(video3d_reader), desc="Converting 3DU data", mininterval=1):

        if frame is None: break;
        # if i % 10 != 0: continue;
        if i >= 300:
            break

        # Read data
        info = frame.info()
        img = frame.img().astype(np.uint8)
        depth = np.squeeze(frame.depth())
        confidence = np.squeeze(frame.confidence())

        # Get parameters
        intrinsics = np.squeeze(np.array(info['cameraIntrinsics'])).T
        extrinsics = np.squeeze(np.array(info['localToWorld'])).T
        extrinsics_inv = np.linalg.inv(extrinsics)

        # Save frame
        img = frame.img().astype(np.uint8)
        fname = f'{frame_id:06d}.png'
        cv2.imwrite(os.path.join(output_images, fname), img)

        # Set camera
        H,W = img.shape[:2]

        # TODO this is to fix the fovx and fovy values, why?
        # Seems to suggest video was downsampled from 1920x1440?
        H*=2
        W*=2

        f_cameras.write(json.dumps({
            "image": fname,
            "fovx": focal2fov(intrinsics[0,0], W),
            "fovy": focal2fov(intrinsics[1,1], H),
            "cx_frac": intrinsics[0,2] / W,
            "cy_frac": intrinsics[1,2] / H,
            "R": extrinsics_inv[:3,:3].tolist(),
            "t": extrinsics_inv[:3,3].tolist(),
        })+"\n")

        # Reszie image to match depth info
        row_ratio = depth.shape[0] / img.shape[0]
        col_ratio = depth.shape[1] / img.shape[1]
        img = cv2.resize(img, (depth.shape[1], depth.shape[0]))

        # Combine color and depth
        color_image = o3d.geometry.Image(np.array(img[:,:,::-1]))

        depth_raw = depth.astype(np.uint16)
        depth_image = np.zeros((depth.shape[0], depth.shape[1]), dtype=np.uint16)
        depth_image[confidence >= 2] = depth_raw[confidence >= 2]
        depth_image = o3d.geometry.Image(depth_image)

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
        if pointcloud is None:
            pointcloud = pcd
        else:
            pointcloud += pcd


        frame_id += 1

    # Downsample to 50K points
    N = len(pointcloud.points)
    if N>50_000:
        pointcloud = pointcloud.random_down_sample( 50_000 / N )

    o3d.io.write_point_cloud(output_ply, pointcloud,write_ascii=True)

    f_cameras.close()

if __name__ == '__main__':
    video3d_file = '/home/jip/data1/3du_data_5/begin_good_day_212/upload/2023_11_01__11_14_30/video.v3dc'
    output_dir = "/home/jip/data1/3du_data_5/gsplat"
    convert_3du(video3d_file, output_dir)
