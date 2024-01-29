import os
import numpy as np


def read_cameras(path:str) -> dict[str,dict]:
    """
    Read cameras.txt file (or equivalent) at `path`,
    assumes the header of the file says exactly:

    ```
    # Camera list with one line of data per camera:
    #   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]
    ```

    Current camera models implemented:
    - PINHOLE: PARAMS[] = [fx, fy]

    Will print a warning for every line that could not be parsed

    Returns a dictionary mapping camera id to a dict with keys
    `id:int`,`model:str`,`width:int`,`height:int`,`fx:float`,`fy:float`
    """
    cameras = {}

    # If path is not a file, `source_path` or `data_folder` is not correctly set
    # in ColmapDataSet
    if not os.path.isfile(path):
        raise ValueError(f"Reading cameras at {path} not possible, file does not exist")

    # Read file
    line_num = 0
    with open(path, 'r') as file:
        while True:

            line = file.readline()
            line_num += 1
            if line is None or not line: break;
            line = line.strip()

            # Skip commented and empty
            if line.startswith('#') or not line : continue;

            # Try to parse line, but only print a warning if a line fails
            try:
                # Split on spaces
                params = line.split()

                # First four parameters are always the same
                (camera_id, model, width, height), params = params[:4], params[4:]

                # Check model
                if model == "PINHOLE":

                    # Export in expected format
                    cameras[int(camera_id)] = {
                        "id": int(camera_id),
                        "model": model,
                        "width": int(width),
                        "height": int(height),
                        "fx": float(params[0]),
                        "fy": float(params[1]),
                    }

                # This will get caught in the clause below to print a warning only
                else:
                    raise NotImplementedError("Camera model is not 'PINHOLE', which is the only model implemented")

            # If something went wrong, simply print line number and error message as a warning
            except Exception as e:
                print(f"\n!Warning! could not parse line number {line_num} in {path}.\n{type(e)}: {e}")
    return cameras


def read_images(path:str) -> list[dict]:
    """
    Read images.txt file (or equivalent) at `path`,
    assumes the header of the file says exactly:

    ```
    # Image list with two lines of data per image:
    #    IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME
    #    POINTS2D[] as (X, Y, POINT3D_ID)
    ```

    Will print a warning for every line that could not be parsed

    POINTS2D[] will be ignored, as we are only interested in image extrinsics

    Returns a list of dictionaries with keys
    `id:int`,`qvec:np.ndarray`,`tvec:np.ndarray`,`camera_id:int`,`name:str`

    `qvec` is in wxyz format
    """
    images = []

    # If path is not a file, `source_path` or `data_folder` is not correctly set
    # in ColmapDataSet
    if not os.path.isfile(path):
        raise ValueError(f"Reading images at {path} not possible, file does not exist")

    # Read file
    line_num = 0
    with open(path, 'r') as file:
        while True:

            line = file.readline()
            line_num += 1
            if line is None or not line: break;
            line = line.strip()

            # Skip commented
            if line.startswith('#'): continue;

            # Try to parse line, but only print a warning if a line fails
            try:

                # Format is fixed, export in expected format
                id,qw,qx,qy,qz,tx,ty,tz,cam_id,name = line.split()

                qvec = np.array([float(qw), float(qx), float(qy), float(qz)])
                qvec /= np.linalg.norm(qvec)

                images.append({
                    "id": int(id),
                    "qvec": qvec,
                    "tvec": np.array([float(tx), float(ty), float(tz)]),
                    "camera_id": int(cam_id),
                    "name": name,
                })

            # If something went wrong, simply print line number and error message as a warning
            except Exception as e:
                print(f"\n!Warning! could not parse line number {line_num} in {path}.\n{type(e)}: {e}")


            # The current line contains camera extrinsics, so skip the next one
            # as it contains POINTS2D[] info
            file.readline()
            line_num+=1

    return images


def read_points3D(path:str) -> tuple[np.ndarray,np.ndarray]:
    """
    Read points3D.txt file (or equivalent) at `path`,
    assumes the header of the file says exactly:

    ```
    # 3D point list with one line of data per point:
    #   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)
    ```

    Will print a warning for every line that could not be parsed

    POINT3D_ID,ERROR,TRACK[] will be ignored

    Returns a tuple `means:np.ndarray[float]`,`colors:np.ndarray[float]`
    of shape `N,3`, where `means` are `x,y,z` coordinates and `colors` are
    `r,g,b` values scaled to be in range `[0,1]`.
    """
    means = []
    colors = []

    # If path is not a file, `source_path` or `data_folder` is not correctly set
    # in ColmapDataSet
    if not os.path.isfile(path):
        raise ValueError(f"Reading points3D at {path} not possible, file does not exist")

    # Read file
    line_num = 0
    with open(path, 'r') as file:
        while True:

            line = file.readline()
            line_num += 1
            if line is None or not line: break;
            line = line.strip()

            # Skip commented
            if line.startswith('#') or not line: continue;

            # Try to parse line, but only print a warning if a line fails
            try:

                _,x,y,z,r,g,b,_ = line.split(maxsplit=7)
                means.append(  np.array( [float(x), float(y), float(z)]) )
                colors.append( np.array( [float(r), float(g), float(b)]) / 255. )

            # If something went wrong, simply print line number and error message as a warning
            except Exception as e:
                print(f"!Warning! could not parse line number {line_num} in {path}.\n{type(e)}: {e}")

    means = np.vstack(means)
    colors = np.vstack(colors)
    return means, colors

if __name__ == '__main__':
    read_points3D('/home/jip/data1/3du_data_2/sparse/0/points3D.txt')
