import numpy as np
from .dataset import DataSet

def get_rotating_dataset(
        num_cameras=120,
        center_point=np.array([0,0,0]),
        rotation_axis=np.array([0,1,0]),
        distance_from_center=10,
        device='cuda',
    ) -> DataSet:
    from scipy.spatial.transform import Rotation
    from src.camera import Camera

    rotvecs = np.linspace(0,2*np.pi,num_cameras).reshape(-1,1) * rotation_axis
    rotmats = Rotation.from_rotvec(rotvecs).as_matrix()

    t = distance_from_center * np.array([0,0,1])

    cameras = []
    for idx in range(num_cameras):
        R = rotmats[idx]
        cameras.append(Camera(
            R = R,
            t = t + center_point @ R.T,
            fovx=np.deg2rad(90),
            fovy=np.deg2rad(60),
            H=1080,W=1920,
            device=device
        ))

    return DataSet(
        cameras=cameras,
    )
