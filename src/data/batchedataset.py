import os
import random
import json

from src.utils.colmap_utils import *
from src.camera import Camera
from src.utils import qvec2rotmat_np,image_path_to_tensor,focal2fov

from threading import Thread

class ThreadWithReturnValue(Thread):
    """From https://stackoverflow.com/a/6894023"""

    def __init__(self, group=None, target=None, name=None,
                 args=(), kwargs={}, Verbose=None):
        super().__init__(group, target, name, args, kwargs)
        self._return = None

    def run(self):
        if self._target is not None:
            self._return = self._target(*self._args,
                                                **self._kwargs)
    def join(self, *args):
        super().join(*args)
        return self._return

class BatchedDataSet:
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
        ):

        source_path = os.path.abspath(source_path)
        cameras_file = os.path.join(source_path, cameras_file)

        cameras = []

        with open(cameras_file,'r') as f:
            line = f.readline()
            idx=1

            while line:
                print(f"\rReading cameras... {idx}",end="",flush=True)

                camera = json.loads(line)

                # Add Camera instance
                cameras.append(camera)

                line = f.readline()
                idx+=1

        print()

        cameras = sorted(cameras, key=lambda cam: cam['image'])

        # Initialize DataSet instance
        return BatchedDataSet(
            cameras=cameras,
            path_images=os.path.join(source_path,images_folder),
            rescale=rescale,
            device=device,
        )


    def __init__(
            self,
            cameras:list[dict[str]],
            path_images:str='images',
            batch_size:int=50,
            rescale:float=1.,
            device='cuda',
        ) -> None:

        self._cameras = cameras

        self._len = len(cameras)
        assert self._len >0, "Cannot initialize empty dataset"
        if self._len < batch_size:
            print("Warning! Batch size is smaller than dataset, "
                  "setting batch size to full set.")

        self.batch_size = min(self._len, batch_size)

        self.path_images = path_images
        self.device = device
        self.rescale = rescale

        # TODO: figure out how to compute this without initializing everything
        self.scene_extend:float = 1.

        # Create train and test split
        self._train_idxs = list(idx for idx in range(self._len) if idx%8 != 0)
        self._test_idxs = list(idx for idx in range(self._len) if idx%8 == 0)


    def __len__(self):
        """
        Total number of Camera instances in this dataset
        """
        return self._len


    def __iter__(self):
        """
        Iterate dataset, equivalent to `DataSet.iter('all')`
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

            assert len(idxs) <= 310

            # Set up batches of indices
            batches = [
                idxs[i:i+self.batch_size]
                for i in range(0, len(self), self.batch_size)
            ]

            # Load first batch
            batch = self.load_batch(batches.pop())

            # Keep going while there are still more batches
            while batches:
                # Start loading next batch
                thread = ThreadWithReturnValue(target=self.load_batch, args=[batches.pop()])
                thread.start()

                # In the meantime, return the current batch
                for cam in batch:
                    yield cam

                # Wait until the next batch is loaded
                batch = thread.join()

            # Dont forget about the last batch!
            for cam in batch:
                yield cam

            # Stop cycling
            if not cycle: break

    def load_camera(self, idx:int) -> Camera:
        camera = self._cameras[idx]

        I = image_path_to_tensor(
            os.path.join(self.path_images, camera['fname']),
            rescale=self.rescale
        )

        return Camera(
            R=np.array(camera['R']),
            t=np.array(camera['t']),
            device=self.device,
            fovx=camera['fovx'],
            fovy=camera['fovy'],
            cx_frac=camera['cx_frac'],
            cy_frac=camera['cy_frac'],
            gt_image=I,
            name=camera['fname'],
            znear=1e-3
        )

    def load_batch(self, idxs:list[int]) -> list[Camera]:
        return [self.load_camera(idx) for idx in idxs]
