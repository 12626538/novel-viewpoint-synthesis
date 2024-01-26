import os
import random
import numpy as np

from ..utils.camera import Camera
from ..utils import image_path_to_tensor,focal2fov

class CustomDataSet:
    """
    Custom dataset
    """
    def __init__(
        self,
        root_dir:str,
        img_folder:str='images',
        data_folder:str='.',
        rescale:float=None,
        device='cuda',
        shuffled:bool=True,
    ) -> None:
        """
        Initialize CustomDataset by camera intrinsics and extrinsics file

        Parameters:
        - `root_dir:str` Root directory of data set
        - `img_folder:str='images/'` Folder with images,
            relative to `root_dir`
        - `data_folder:str='.'` Folder with `inv_instriniscs_array.npy` and
            `pose_array.npy`, relative to `root_dir`
        - `rescale:Optional[float]` Downsample images to `1/rescale` scale.
            Will try to see if `[root_dir]/[img_folder]_[rescale]/` exists
            and read images from there
        - `shuffled:bool=True` Whether to shuffle the cameras when iterating
            this instance

        Raises ValueError if `inv_intrinsic_array.npy` and `pose_array.npy`
        cannot be found in `[root_dir]/[data_folder]`

        Raises FileNotFoundError if images are not found
        in `[root_dir]/[img_folder]/[idx].png`
        where `[idx]` is the row index of the camera intrinsics array
        """

        self.device = device
        self.shuffled = shuffled

        root_dir = os.path.abspath(root_dir)
        data_folder = os.path.join(root_dir, data_folder)
        img_folder = os.path.join(root_dir, img_folder)

        # Read instrinsics and extrinsics
        cam_intrs = np.load(os.path.join(data_folder, 'inv_intrinsic_array.npy'))
        cam_extrs = np.load(os.path.join(data_folder, 'pose_array.npy'))

        # Sanity check, intrinsics and extrinsics should be same number of images
        assert cam_intrs.shape[0] == cam_extrs.shape[0], "Camera intrinsics and poses do not share the same dim length"

        # See if `[root_dir]/[image_folder]_[rescale]/` exists
        if os.path.isdir("{}_{}".format(img_folder.rstrip('/'), rescale)):
            # Use that folder instead, dont rescale images
            img_folder = "{}_{}".format(img_folder.rstrip('/'), rescale)
            rescale=None

        self.cameras:list[Camera] = []
        for idx in range(cam_intrs.shape[0]):
            print(f"\rParsing cameras and images... {idx}/{cam_intrs.shape[0]}",end="",flush=True)
            fname = f"{idx:06d}.png"

            self.cameras.append(Camera(
                gt_image=image_path_to_tensor( os.path.join(img_folder, fname) ),
                R=cam_extrs[idx, :3,:3],
                t=cam_extrs[idx, :3,3],
                fovx=focal2fov(797.78, 944),
                fovy=focal2fov(805.42, 705),
                name=fname,
                device=self.device
            ))
        print()

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
        Create infinite iterator cycling through cameras

        This method will asure all cameras are iterated at least once
        before any will be yielded a second time.

        If `ColmapDataSet.shuffled` is set to `True`, cameras will be yielded
        in random order, and is reshuffled every time all cameras are iterated
        unlike `itertools.cycle(dataset)`.
        """
        while True:
            for camera in self:
                yield camera
