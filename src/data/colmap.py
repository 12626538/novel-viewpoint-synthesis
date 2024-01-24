import os
import random

from ..utils.colmap_utils import *
from ..utils.camera import Camera

class ColmapDataSet:
    def __init__(
        self,
        root_dir:str,
        img_folder:str='images',
        data_folder:str='sparse/0',
        rescale:float=None,
        device='cuda:0',
        shuffled:bool=True,
    ) -> None:
        """
        Initialize ColmapDataSet

        - @parameter `root_dir:str` Root directory of data set
        - @parameter `img_folder:str='images/'` Folder with images,
            relative to `root_dir`
        - @parameter `data_folder:str='sparse/0/'` Folder with data files,
            such as `cameras.txt` and `images.txt`
        - @parameter `rescale:Optional[float]` Downsample images to `1/rescale` scale.
            Will try to see if `[root_dir]/[img_folder]_[rescale]/` exists and read images from there

        Raises ValueError if `cameras.txt`, `images.txt` and `points3D.txt`
        cannot be found in `[root_dir]/[data_folder]`

        Raises FileNotFoundError if images in `images.txt` are not found
        in `[root_dir]/[img_folder]/[name]`
        """

        self.device = device
        self.shuffled = shuffled

        root_dir = os.path.abspath(root_dir)
        data_folder = os.path.join(root_dir, data_folder)
        img_folder = os.path.join(root_dir, img_folder)

        # Read COLMAP cameras intrinsics and image extrinsics
        cameras = read_cameras( os.path.join(data_folder, 'cameras.txt') )
        images = read_images( os.path.join(data_folder, 'images.txt') )

        # See if `[root_dir]/[image_folder]_[rescale]/` exists
        if os.path.isdir("{}_{}".format(img_folder.rstrip('/'), rescale)):
            # Use that folder instead, dont rescale images
            img_folder = "{}_{}".format(img_folder.rstrip('/'), rescale)
            rescale=None

        print("Parsing cameras and images...")
        self.cameras:list[Camera] = []
        for image in images:
            camera = cameras[image['camera_id']]

            self.cameras.append(Camera(
                gt_image=image_path_to_tensor( os.path.join(img_folder, image['name']) ),
                R=qvec2rotmat(image['qvec']).T,
                t=image['tvec'],
                fovx=focal2fov(camera['fx'], camera['width']),
                fovy=focal2fov(camera['fy'], camera['height']),
                name=image['name'],
                device=self.device
            ))

        self.cameras = sorted(self.cameras, key=lambda cam:cam.name)


        camera_positions = np.vstack([camera.t for camera in self.cameras])
        center_camera = camera_positions.mean(axis=0,keepdims=True)

        self.scene_extend:float = np.linalg.norm(camera_positions - center_camera, axis=0).max()


    def __len__(self):
        return len(self.cameras)

    def __iter__(self):
        """
        Iterate dataset
        """
        _idxs = list(range(len(self)))
        if self.shuffled:
            random.shuffle(_idxs)

        for id in _idxs:
            yield self.cameras[id]

    def cycle(self):
        """
        Like iter, but loops infinitely
        """
        while True:
            for camera in self:
                yield camera


if __name__ == '__main__':
    db = ColmapDataSet(
        '/home/jip/data1/3du_data_2/',
        device='cuda:0' if torch.cuda.is_available() else 'cpu'
    )

    print(f"Found {len(db)} cameras")
