import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Line3DCollection
import numpy as np
import os

from src.data import DataSet
from convert import convert

def e2h(pts):
    return np.hstack((pts,np.ones((pts.shape[0],1))))

def h2e(pts):
    return pts[:,:3] / pts[:,3:4]

ROOT_DIR = '/home/jip/Desktop/tmp/3du_data_2'

# convert(ROOT_DIR)
data = DataSet.from_intr_extr(ROOT_DIR,rescale=8,device='cpu')

fig = plt.figure(figsize=(10,5), tight_layout=True)
ax:plt.Axes = fig.add_subplot(1,2,1,projection='3d')
ax2:plt.Axes = fig.add_subplot(1,2,2)

pts = np.array([
    [ 0,  0,  0],
    [ 1,  0,  0],
    [ 0,  1,  0],
    [ 0,  0,  1],
])
edges = np.array([
    [0,0,0,],
    [1,2,3,]
])


s=.3
S = np.array([
    [s, 0, 0, 0,],
    [0, s, 0, 0,],
    [0, 0, s, 0,],
    [0, 0, 0, 1,],
])

ax.quiver(0,0,0, 1,0,0, color='r')
ax.set_xlabel('x')
ax.quiver(0,0,0, 0,0,1, color='g')
ax.set_zlabel('y')
ax.quiver(0,0,0, 0,1,0, color='b')
ax.set_ylabel('z')

scene_scale = 1.5
ax.set_xlim(-scene_scale,  scene_scale)
ax.set_ylim( scene_scale, -scene_scale)
ax.set_zlim(-scene_scale,  scene_scale)
ax.set_aspect('equal')

image = None
lines1 = None
lines2 = None

for idx,cam in enumerate(data.cameras):

    I = cam.gt_image.permute(1,2,0).numpy()
    H,W=I.shape[:2]

    if image is None:
        image = ax2.imshow(I)
    else:
        image.set_data(I)

    C2W = np.linalg.inv(cam.viewmat.detach().numpy())

    vertices = h2e( e2h(pts) @ S.T @ C2W.T )
    vertices[:,[1,2]] = vertices[:,[2,1]]

    segments = np.stack( (vertices[edges[0]], vertices[edges[1]]), axis=1)

    if lines1 is None:
        lines1 = Line3DCollection(segments, colors=['red','blue','green'])
        ax.add_collection3d(lines1)
    else:
        lines1.set_segments(segments)

    plt.pause(1/30)
