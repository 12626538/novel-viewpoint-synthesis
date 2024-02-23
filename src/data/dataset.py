import os
import random
import json

from src.utils.colmap_utils import *
from src.camera import Camera
from src.utils import qvec2rotmat_np,image_path_to_tensor,focal2fov

class DataSet:
    """
    DataSet super class. Is mostly just a fancy wrapper for a list of
    `Camera` instances.

    By default, `Camera` instances are split such that every 8th camera
    is considered a test view, and the remaining cameras are views used for
    training.

    Usage: Iterate cameras in order (or shuffled if `shuffled=True`)
    >>> dataset = DataSet(cameras, shuffled=False)
    >>> for camera in dataset:
    >>>     print(camera.name)

    Usage: similar to above, but keep cycling the dataset infinitely
    >>> dataset = ColmapDataSet(...)
    >>> ds_iter = dataset.cycle() # Create infinte iterator
    >>> while True:
    >>>     camera = next(ds_iter)
    """

    @classmethod
    def from_colmap(
            cls,
            source_path:str,
            images_folder:str='images',
            data_folder:str='sparse/0',
            rescale:int=1,
            device='cuda',
            **kwargs,
        ) -> 'DataSet':
        """
        Construct DataSet instance from COLMAP dataset

        Parameters:
        - `source_path:str` Root directory of data set
        - `images_folder:str='images/'` Folder with images, relative to
            `source_path`
        - `data_folder:str='sparse/0/'` Folder with data files,
            such as `cameras.txt` and `images.txt`
        - `rescale:Optional[int]` Downsample images to `1/rescale` scale.
            Will try to see if `[source_path]/[images_folder]_[rescale]/` exists
            and read images from there

        Raises ValueError if `cameras.txt` or `images.txt`  cannot be found
        in `[source_path]/[data_folder]`

        Raises FileNotFoundError if images specified in `images.txt` are
        not found in `[source_path]/[images_folder]/[name]`
        """

        source_path = os.path.abspath(source_path)
        data_folder = os.path.join(source_path, data_folder)
        images_folder = os.path.join(source_path, images_folder)

        # Read COLMAP cameras intrinsics and image extrinsics
        intrinsics = read_cameras( os.path.join(data_folder, 'cameras.txt') )
        extrinsics = read_images( os.path.join(data_folder, 'images.txt') )

        # See if `[source_path]/[image_folder]_[rescale]/` exists
        newfolder = "{}_{}".format(images_folder.rstrip('/'), rescale)
        if os.path.isdir(newfolder):
            # Use that folder instead, dont rescale images
            images_folder = newfolder
            rescale = 1

        # Iterate all images
        N = len(extrinsics)
        cameras = []
        for idx,extr in enumerate(extrinsics,1):
            print("\rParsing cameras and images... {:4}/{:4}".format(idx,N),flush=True,end="")

            # Get camera info associated with this image
            intr = intrinsics[extr['camera_id']]

            # Add Camera instance to dataset
            cameras.append(Camera(
                gt_image=image_path_to_tensor( os.path.join(images_folder, extr['name']), rescale=rescale ),
                R=qvec2rotmat_np(extr['qvec']),
                t=extr['tvec'],
                fovx=focal2fov(intr['fx'], intr['width']),
                fovy=focal2fov(intr['fy'], intr['height']),
                name=extr['name'],
                device=device
            ))
        print()

        # Initialize DataSet instance
        return DataSet(
            cameras=cameras,
            **kwargs,
        )


    @classmethod
    def from_3du(
            cls,
            source_path:str,
            cameras_file:str='cameras.json',
            images_folder:str='images',
            rescale:int=1,
            device='cuda',
            data_folder=None,# IGNORED
            **kwargs,
        ) -> 'DataSet':
        """
        Construct DataSet instance from intrinsics and extrinsics file

        Parameters:
        - `source_path:str` Root directory of data set
        - `cameras_file:str='camearas.json'` Camera intrinsics,
            relative to `source_path`
        - `images_folder:str='images/'` Folder with images, relative to
            `source_path`
        - `rescale:Optional[int]` Downsample images to `1/rescale` scale.
            Will try to see if `[source_path]/[images_folder]_[rescale]/` exists
            and read images from there
        """

        source_path = os.path.abspath(source_path)
        cameras_file = os.path.join(source_path, cameras_file)

        cameras = []

        with open(cameras_file,'r') as f:
            line = f.readline()
            idx=1

            while line:
                print(f"\rReading cameras... {idx}",end="",flush=True)

                camera = json.loads(line)

                # Get ground truth image
                I = image_path_to_tensor( os.path.join(source_path, images_folder, camera['image']), rescale=rescale)

                # Add Camera instance
                cameras.append(Camera(
                    R=np.array(camera['R']),
                    t=np.array(camera['t']),
                    device=device,
                    fovx=camera['fovx'],
                    fovy=camera['fovy'],
                    cx_frac=camera['cx_frac'],
                    cy_frac=camera['cy_frac'],
                    gt_image=I,
                    name=camera['image'],
                ))

                line = f.readline()
                idx+=1

        print()

        # Initialize DataSet instance
        return DataSet(
            cameras=cameras,
        )


    def __init__(
            self,
            cameras:list[Camera],
        ) -> None:
        """
        Initialize DataSet instance from a list of cameras

        Parameters:
        - `cameras:list[Camera]` A list of cameras to use
        """

        # Make sure cameras appear in the same order every run
        self.cameras = sorted(cameras, key=lambda cam:cam.name)

        # Compute scene center
        camera_positions = np.vstack([camera.loc for camera in self.cameras])
        center_camera = camera_positions.mean(axis=0,keepdims=True)

        # Compute size s.t. all cameras fit in range [-scene_extend, scene_extend] in X,Y,Z directions
        self.scene_extend:float = 1.1 * np.linalg.norm(camera_positions - center_camera, axis=1).max()

        # Create train and test split
        self._train_idxs = list(idx for idx in range(len(self.cameras)) if idx%8 != 0)
        self._test_idxs = list(idx for idx in range(len(self.cameras)) if idx%8 == 0)

    def __len__(self):
        """
        Total number of Camera instances in this dataset
        """
        return len(self.cameras)


    def __iter__(self):
        """
        Iterate dataset
        """
        return self.iter(partition="all")


    def iter(self, partition="train", cycle=False, shuffle=False):
        """
        Iterate dataset with additional options

        Parameters:
        - `partition:Literal['train'|'test'|'all']` - What partition of the data
            to use. Default is trainset
        - `cycle:bool` - Whether to keep iterating infinitily. Differs from
            `itertools.cycle` in that - if `shuffle=True` - the dataset will
            reshuffle itself after yielding all instances once. Default is False
        - `shuffle:bool` - Whether to yield instances in a random order.
        """
        # Keep cycling...
        while True:

            # Get camera indices for selected set
            if partition == "train":
                idxs = self._train_idxs.copy()
            elif partition=="test":
                idxs = self._test_idxs.copy()
            else:
                idxs = list(range(len(self)))

            # Shuffle data
            if shuffle: random.shuffle(idxs)

            # Iterate cameras
            for idx in idxs:
                yield self.cameras[idx]

            # Stop cycling
            if not cycle: break


    def oneup_scale(self):
        """
        Update each camera's ground truth image with an image a scale higher than
        the current.

        For example, if the current scale is 8, it replaces all gt images with a
        scale 4 version.
        """
        raise DeprecationWarning()
        # Update current scale, with a fallback of scale 1
        mapper = { 8:4, 4:2, 2:1, 1:1 }
        self.current_scale = mapper.get(self.current_scale, 1)

        # Rescale images to this scale
        rescale = self.current_scale

        # See if `[source_path]/[image_folder]_[scale]/` exists
        images_folder = self.image_folder
        newfolder = "{}_{}".format(images_folder.rstrip('/'), self.current_scale)
        if os.path.isdir(newfolder):
            # Use that folder instead, dont rescale images
            images_folder = newfolder
            rescale = 1

        for cam in self.cameras:
            cam.set_gt_image(image_path_to_tensor( os.path.join(images_folder, cam.name), rescale=rescale ))
