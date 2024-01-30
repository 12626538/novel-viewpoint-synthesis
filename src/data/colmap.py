import os
import random

from ..utils.colmap_utils import *
from ..utils.camera import Camera
from ..utils import qvec2rotmat,image_path_to_tensor,focal2fov

class ColmapDataSet:
    """
    COLMAP dataset class. Is mostly just a fancy wrapper for a list of
    `Camera` instances.

    Usage: Iterate cameras in order (or shuffled if `shuffled=True`)
    >>> dataset = ColmapDataSet(..., shuffled=False)
    >>> for camera in dataset:
    >>>     print(camera.name)

    Usage: similar to above, but keep cycling the dataset infinitely
    >>> dataset = ColmapDataSet(...)
    >>> ds_iter = dataset.cycle() # Create infinte iterator
    >>> while True:
    >>>     camera = next(ds_iter)
    """
    def __init__(
        self,
        source_path:str,
        images_folder:str='images',
        data_folder:str='sparse/0',
        rescale:int=1,
        device='cuda',
        shuffled:bool=True,
    ) -> None:
        """
        Initialize ColmapDataSet

        Parameters:
        - `source_path:str` Root directory of data set
        - `images_folder:str='images/'` Folder with images,
            relative to `source_path`
        - `data_folder:str='sparse/0/'` Folder with data files,
            such as `cameras.txt` and `images.txt`
        - `rescale:Optional[int]` Downsample images to `1/rescale` scale.
            Will try to see if `[source_path]/[images_folder]_[rescale]/` exists
            and read images from there
        - `shuffled:bool=True` Whether to shuffle the cameras when iterating
            this instance

        Raises ValueError if `cameras.txt`, `images.txt` and `points3D.txt`
        cannot be found in `[source_path]/[data_folder]`

        Raises FileNotFoundError if images in `images.txt` are not found
        in `[source_path]/[images_folder]/[name]`
        """

        assert rescale in {1,2,4,8}

        self.device = device
        self.shuffled = shuffled

        source_path = os.path.abspath(source_path)
        data_folder = os.path.join(source_path, data_folder)
        images_folder = os.path.join(source_path, images_folder)

        self.source_path = source_path
        self.data_folder = data_folder
        self.image_folder = images_folder

        # Read COLMAP cameras intrinsics and image extrinsics
        cameras = read_cameras( os.path.join(data_folder, 'cameras.txt') )
        images = read_images( os.path.join(data_folder, 'images.txt') )

        self.current_scale = rescale

        # See if `[source_path]/[image_folder]_[rescale]/` exists
        newfolder = "{}_{}".format(images_folder.rstrip('/'), rescale)
        if os.path.isdir(newfolder):
            # Use that folder instead, dont rescale images
            images_folder = newfolder
            rescale = 1


        N = len(images)
        self.cameras:list[Camera] = []
        for idx,image in enumerate(images,1):
            print("\rParsing cameras and images... {:4}/{:4}".format(idx,N),flush=True,end="")
            camera = cameras[image['camera_id']]

            self.cameras.append(Camera(
                gt_image=image_path_to_tensor( os.path.join(images_folder, image['name']), rescale=rescale ),
                R=qvec2rotmat(image['qvec']).T,
                t=image['tvec'],
                fovx=focal2fov(camera['fx'], camera['width']),
                fovy=focal2fov(camera['fy'], camera['height']),
                name=image['name'],
                device=self.device
            ))
        print()

        # Make sure cameras appear in the same order every run
        self.cameras = sorted(self.cameras, key=lambda cam:cam.name)

        # Compute scene center
        camera_positions = np.vstack([camera.t for camera in self.cameras])
        center_camera = camera_positions.mean(axis=0,keepdims=True)

        # Compute size s.t. all cameras fit in range [-scene_extend, scene_extend] in X,Y,Z directions
        self.scene_extend:float = 1.1 * np.linalg.norm(camera_positions - center_camera, axis=1).max()

        # Move all cameras into this range
        # center_camera=center_camera.flatten()
        # for cam in self.cameras:
        #     cam.t -= center_camera

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


    def oneup_scale(self):
        # Update current scale, with a fallback of scale 1 in case of error
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
