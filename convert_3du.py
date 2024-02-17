import os
import numpy as np
import open3d as o3d
import video3d
import cv2

import tqdm


def create_point_cloud(
        video3d_file:str,
        output_dir:str,
        dname_images:str='images',
        fname_ply:str='pointcloud.ply',
        fname_intr:str='intrinsics.npy',
        fname_extr:str='poses.npy',
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
    output_intr = os.path.join(output_dir, fname_intr)
    output_extr = os.path.join(output_dir, fname_extr)

    output_images = os.path.join(output_dir, dname_images)
    os.makedirs(output_images, exist_ok=True)

    print("Writing to", output_dir)

    # Set up video reader
    video3d_reader = video3d.Video3DReader(video3d_file, video3d.READ_ALL)

    # Set up output variables
    point_clouds = []
    intrinsics = []
    extrinsics = []
    frame_id = 0
    for i, frame in tqdm.tqdm(enumerate(video3d_reader)):

        if frame is None: break;
        # if i % 10 != 0: continue;
        if i >= 300:
            break

        # Read data
        info:dict = frame.info()
        img:np.ndarray = frame.img().astype(np.uint8)
        depth:np.ndarray = np.squeeze(frame.depth())
        confidence = np.squeeze(frame.confidence())

        camera_intrinsics = np.squeeze(np.array(info['cameraIntrinsics'])).T
        intrinsics.append(camera_intrinsics)

        camera_extrinsics = np.squeeze(np.array(info['localToWorld'])).T
        extrinsics.append(camera_extrinsics)

        # from src.camera import Camera
        # from src.utils import focal2fov
        # import torch

        # print(list(info.keys()))
        # intr = camera_intrinsics
        # extr = np.linalg.inv(camera_extrinsics)
        # H,W = img.shape[:2]

        # VP = np.array(info['viewProjectionMatrix']).squeeze()
        # cam = Camera(
        #     R=extr[:3,:3],
        #     t=extr[:3,3],
        #     device='cpu',
        #     fovx=focal2fov(intr[0,0], W)*2,
        #     fovy=focal2fov(intr[1,1], H)*2,
        #     gt_image=torch.from_numpy(img).permute(2,0,1),
        # )
        # print(intr)
        # print(np.rad2deg(cam.fovx), np.rad2deg(cam.fovy))
        # return

        # Save frame
        fname = os.path.join(output_images, f'{frame_id:06d}.png')
        cv2.imwrite(fname, img)

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
        fx = camera_intrinsics[0, 0] * col_ratio * 0.5
        fy = camera_intrinsics[1, 1] * row_ratio * 0.5
        cx = camera_intrinsics[0, 2] * col_ratio * 0.5
        cy = camera_intrinsics[1, 2] * row_ratio * 0.5
        width = depth.shape[1]
        height = depth.shape[0]
        o3d_camera_intrinsics = o3d.camera.PinholeCameraIntrinsic(width, height, fx, fy, cx, cy)

        # Get pointcloud in 3D camera coordinates
        pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
            rgbd_image,
            o3d_camera_intrinsics,
        )


        # # Convert points to world coordinates
        pcd.transform(camera_extrinsics)

        # Save results
        point_clouds.append(pcd)


        frame_id += 1

    # Save in- and extrinsics as a big numpy array
    intrinsics = np.stack(intrinsics, axis=0)
    with open(output_intr, 'wb') as f:
        np.save(f, intrinsics)

    extrinsics = np.stack(extrinsics, axis=0)
    with open(output_extr, 'wb') as f:
        np.save(f, extrinsics)

    # Combine point cloud of all frames
    combined_point_cloud = point_clouds[0]
    for pc in point_clouds[1:]:
        combined_point_cloud += pc

    # Downsample to 50K points
    N = len(combined_point_cloud.points)
    if N>50_000:
        combined_point_cloud = combined_point_cloud.random_down_sample( 50_000 / N )

    o3d.io.write_point_cloud(output_ply, combined_point_cloud,write_ascii=True)


# video3d_file = "../data1/data/begin_good_day_212/upload/2023_11_01__11_14_30/video.v3dc"
video3d_file = '/home/jip/data1/3du_data_5/begin_good_day_212/upload/2023_11_01__11_14_30/video.v3dc'
output_dir = "/home/jip/data1/3du_data_5/gsplat"
create_point_cloud(video3d_file, output_dir)
