import os
import numpy as np
import open3d as o3d
import video3d
import cv2


def create_point_cloud(
        video3d_file,
        output_dir,
        dname_images='images',
        fname_ply='pointcloud.ply',
        fname_intr='inv_intrinsic_array.npy',
        fname_extr='pose_array.npy',
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
    for i, frame in enumerate(video3d_reader):

        if frame is None: break;
        if i % 10 != 0: continue;

        # Read data
        info = frame.info()
        img = frame.img()
        depth = np.squeeze(frame.depth())
        confidence = np.squeeze(frame.confidence())

        # Reszie image to match depth info
        row_ratio = depth.shape[0] / img.shape[0]
        col_ratio = depth.shape[1] / img.shape[1]
        img = cv2.resize(img, (depth.shape[1], depth.shape[0]))

        # Combine color and depth
        color_image = o3d.geometry.Image(img.astype(np.uint8))
        depth_raw = depth.astype(np.uint16)
        depth_image = np.zeros((depth.shape[0], depth.shape[1]), dtype=np.uint16)
        depth_image[confidence >= 2] = depth_raw[confidence >= 2]
        depth_image = o3d.geometry.Image(depth_image)
        rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
            color_image, depth_image, depth_scale=1000, depth_trunc=5.0)

        # Get intrinsics
        camera_intrinsics = np.squeeze(np.array(info['cameraIntrinsics'])).T
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

        # Convert points to world coordinates
        local_to_world_matrix = np.squeeze(np.array(info['localToWorld'])).T
        pcd.transform(local_to_world_matrix)

        # Save results
        intrinsics.append(np.squeeze(np.array(o3d_camera_intrinsics)))
        extrinsics.append(local_to_world_matrix)
        point_clouds.append(pcd)

        # Save frame
        fname = os.path.join(output_images, f'{frame_id:06d}.png')
        cv2.imwrite(fname, img.astype(np.uint8))

        frame_id += 1

    # Save in- and extrinsics as a big numpy array
    intrinsics = np.concatenate(intrinsics, axis=0)
    with open(output_intr, 'wb') as f:
        np.save(f, intrinsics)

    extrinsics = np.concatenate(extrinsics, axis=0)
    with open(output_extr, 'wb') as f:
        np.save(f, extrinsics)

    # Combine point cloud of all frames
    combined_point_cloud = point_clouds[0]
    for pc in point_clouds[1:]:
        combined_point_cloud += pc


    # Extracting coordinates from the point cloud
    coords = np.asarray(combined_point_cloud.points) #* 1000

    # Applying transformations
    # coords[:, 0] = np.round(coords[:, 0] / 10) * 10
    # coords[:, 1] = np.round(coords[:, 1] / 100) * 100
    # coords[:, 2] = np.round(coords[:, 2] / 10) * 10
    unique_coords, unique_ind = np.unique(coords, return_index=True, axis=0)
    coords = coords[unique_ind]

    # Updating the point cloud
    combined_point_cloud.points = o3d.utility.Vector3dVector(coords)

    o3d.io.write_point_cloud(output_ply, combined_point_cloud)


# video3d_file = "../data1/data/begin_good_day_212/upload/2023_11_01__11_14_30/video.v3dc"
video3d_file = '/home/jesse/data1/data/begin_good_day_212/process/floor_0/video.v3dc'
output_dir = "/home/jesse/output/created_pcs/sendtojippls/"
create_point_cloud(video3d_file, output_dir)

print("To copy this to Jip:")
print(f"scp -r {output_dir} jip@deepserver3.3duniversum.com:/home/jip/data1/jesse/")
