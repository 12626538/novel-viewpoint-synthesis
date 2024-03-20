import numpy as np

def viewmat_from_loc_lookat(loc:np.ndarray, lookat:np.ndarray) -> np.ndarray:
    R = rotmat_from_lookat(lookat)

    P = np.eye(4)
    P[:3,:3] = R
    P[:3,3] = -loc @ R.T

    return P

def rotmat_from_lookat(lookat, up=np.array([0,-1,0])):
    # Z-axis is lookat
    vec2 = lookat / np.linalg.norm(lookat)

    # X-axis is perp. to Z and up axis
    vec0 = np.cross(up, vec2)
    vec0 /= np.linalg.norm(vec0)

    # Y axis is perp to X,Z axes
    vec1 = np.cross(vec2, vec0)
    vec1 /= np.linalg.norm(vec1)

    return np.stack([vec0,vec1,vec2],1).T


def catmull_rom_spline(P0,P1,P2,P3, num_points, alpha) -> np.ndarray:
    """
    From https://en.wikipedia.org/wiki/Centripetal_Catmull%E2%80%93Rom_spline

    Compute the points in the spline segment
    :param P0, P1, P2, and P3: The (x,y) point pairs that define the Catmull-Rom spline
    :param num_points: The number of points to include in the resulting curve segment
    :param alpha: 0.5 for the centripetal spline, 0.0 for the uniform spline, 1.0 for the chordal spline.
    :return: The points
    """

    # Calculate t0 to t4. Then only calculate points between P1 and P2.
    # Reshape linspace so that we can multiply by the points P0 to P3
    # and get a point for each value of t.
    def tj(ti: float, pi: tuple, pj: tuple) -> float:
        xi, yi, zi = pi
        xj, yj, zj = pj
        dx, dy, dz = xj - xi, yj - yi, zj - zi
        l = (dx ** 2 + dy ** 2 + dz ** 2) ** 0.5
        return ti + l ** alpha

    t0: float = 0.0
    t1: float = tj(t0, P0, P1)
    t2: float = tj(t1, P1, P2)
    t3: float = tj(t2, P2, P3)
    t = np.linspace(t1, t2, num_points).reshape(num_points, 1)

    A1 = (t1 - t) / (t1 - t0) * P0 + (t - t0) / (t1 - t0) * P1
    A2 = (t2 - t) / (t2 - t1) * P1 + (t - t1) / (t2 - t1) * P2
    A3 = (t3 - t) / (t3 - t2) * P2 + (t - t2) / (t3 - t2) * P3
    B1 = (t2 - t) / (t2 - t0) * A1 + (t - t0) / (t2 - t0) * A2
    B2 = (t3 - t) / (t3 - t1) * A2 + (t - t1) / (t3 - t1) * A3
    points = (t2 - t) / (t2 - t1) * B1 + (t - t1) / (t2 - t1) * B2
    return points

def smooth_directions(directions:np.ndarray, num_points:int, alpha:float):
    """
    Given a collection of 3D vectors, generate a catmull-rom spline between
    all points, resulting in `num_point` positions.
    """
    # Compute the number of points per spline
    num_points = round( num_points / (len(directions)-3) )

    # Concatenate catmull-rom splines between every set of 4 points.
    return np.concatenate(
        [ catmull_rom_spline(*directions[i:i+4], num_points=num_points, alpha=alpha)
         for i in range(len(directions)-3)
        ]
    )

def smooth_camera_path(cameras, num_poses=600, alpha:float=1) -> list[np.ndarray]:
    """
    Interpolate between cameras to create a smooth path

    Args:
    - `cameras:list[Camera]` A sorted list of camera instances
    - `num_poses:int` Total number of poses to interpolate
    - `alpha:float` Smoothing factor for Catmull Rom spline function

    Returns:
    - `poses:np.ndarray` A list of np arrays where the ith entry is the 4x4
        interpolated world-to-view matrix
    """
    # Use 3D 'lookAt' vector for direction
    lookat = np.array([0,0,1])

    # get lookAt for every camera
    lookats = np.array([cam.R.T @ lookat for cam in cameras])

    lookats[:,1] /= 2
    lookats /= np.linalg.norm(lookats, axis=-1, keepdims=True)

    n_combine = 10

    # Group take the mean of every N cameras, extrapolate additional start and end value
    lookats = lookats[:(lookats.shape[0]//n_combine)*n_combine].reshape(-1,n_combine,3).mean(axis=1)
    lookats = np.pad(lookats, ((1,1),(0,0)),'reflect',reflect_type='odd')

    # Do the same for camera locations
    locs = np.array([cam.loc for cam in cameras])
    locs = locs[:(locs.shape[0]//n_combine)*n_combine].reshape(-1,n_combine,3).mean(axis=1)
    locs = np.pad(locs, ((1,1),(0,0)),'reflect',reflect_type='odd')

    # Smooth everything out
    smooth_lookats = smooth_directions(lookats, num_poses, alpha)
    Rs = map(rotmat_from_lookat, smooth_lookats)
    smooth_locs = smooth_directions(locs, num_poses, alpha)

    # Reconstruct matrices
    def viewmat(tpl):
        R,loc = tpl
        pose = np.eye(4)
        pose[:3,:3] = R
        pose[:3,3] = -loc@R.T
        return pose
    poses = list(map(viewmat, zip(Rs,smooth_locs)))

    return poses
