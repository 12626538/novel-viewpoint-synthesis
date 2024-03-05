import argparse
import os
import numpy as np
import cv2

import torch
from torchvision.utils import save_image

from src.model.gaussians import Gaussians,RenderPackage
from src.data import DataSet,get_rotating_dataset
from src.arg import ModelParams,DataParams,PipeLineParams,get_args
from src.camera import Camera
from scipy.spatial.transform import Rotation

import time

AXIS_X,AXIS_Y,AXIS_Z = np.eye(3)
LEFT = -AXIS_X
FORWARD = AXIS_Z
UP = -AXIS_Y
SPEED = .1

LOOK_LEFT = Rotation.from_rotvec(AXIS_Y * np.pi/8 * SPEED).as_matrix()
LOOK_UP = Rotation.from_rotvec(-AXIS_X * np.pi/8 * SPEED).as_matrix()

TILT_LEFT = Rotation.from_rotvec(AXIS_Z * np.pi/16).as_matrix()

dragging_start = None
dragging_current = (0,0)
R_drag = np.eye(3)
zsign=-1

mouse_offset = np.zeros(2)

def cb_mouse(event, x, y, flags, param):
    # grab references to the global variables
    global dragging_start,mouse_offset
    # if the left mouse button was clicked, record the starting
    # (x, y) coordinates and indicate that cropping is being
    # performed

    dragging_current = np.array((x,y))
    if event == cv2.EVENT_LBUTTONDOWN:
        dragging_start = np.array((x,y))
    # elif event == cv2.EVENT_LBUTTONUP:
    #     dragging_start = None

    if dragging_start is not None:
        mouse_offset = (dragging_current - dragging_start) / np.array((args.W,args.H))



parser = argparse.ArgumentParser("Training Novel Viewpoint Synthesis")
parser.add_argument("-H", type=int, default=720)
parser.add_argument("-W", type=int, default=1280)
args, (data_args, model_args, pipeline_args) = get_args(DataParams,ModelParams,PipeLineParams, parser=parser)

try:
    torch.cuda.set_device(args.device)
except:
    print(f"!Warning! Could not use device {args.device}, falling back to cuda:0.")
    args.device = 'cuda:0'

# Use checkpoint
if not os.path.exists(pipeline_args.load_checkpoint):
    print(f"Checkpoint {pipeline_args.load_checkpoint} not found")
    exit(1)

print(f"Loading .ply model from {pipeline_args.load_checkpoint}")
model = Gaussians.from_ply(
    device=args.device,
    path_to_ply_file=pipeline_args.load_checkpoint,
    **vars(model_args)
)

model.sh_degree_current = model.sh_degree_max


# loc = np.array([0.45,-1.23,0.6])
loc = np.zeros(3)
glob_scale = 1
bg = torch.tensor([0.,0.,0.],dtype=torch.float32, device=args.device)
rot = np.array([np.pi, 0])

cv2.namedWindow("image")
# cv2.setMouseCallback("image", cb_mouse)

_frame = 0
_tstart = time.time()
framerate = 0
R=np.eye(3)
while True:

    _frame += 1
    if _frame >= 50:
        framerate = round(_frame / (time.time() - _tstart))
        _frame = 0
        _tstart = time.time()

    rot += mouse_offset * -zsign
    R = Rotation.from_rotvec(-AXIS_X * rot[1]).as_matrix() @ Rotation.from_rotvec(AXIS_Y * rot[0]).as_matrix()
    camera = Camera(
        R = R,
        t = -loc @ R.T,
        fovx=np.deg2rad(30),
        fovy=np.deg2rad(30),
        H=args.H,W=args.W,
        device=args.device,
        zsign=zsign
    )
    render = (model.render(camera, glob_scale=glob_scale, bg=bg).rendering.permute(1,2,0).detach().cpu().numpy()*255).astype(np.uint8)

    # RGB -> BGR
    render[:,:,[2,1,0]] = render[:,:,[0,1,2]]

    cv2.putText(render, f"Framerate: {framerate:3d}", [10,40], cv2.FONT_HERSHEY_PLAIN, .7, (0,0,0), 1, cv2.LINE_AA)
    cv2.imshow('image',render)
    key = cv2.waitKey(0)


    if key == 27: break # ESC

    elif key == ord('a'): loc -= LEFT * SPEED @ R
    elif key == ord('w'): loc += FORWARD * SPEED @ R
    elif key == ord('d'): loc += LEFT * SPEED @ R
    elif key == ord('s'): loc -= FORWARD * SPEED @ R

    elif key == ord(' '): loc += UP * SPEED * zsign
    elif key == ord('z'): loc -= UP * SPEED * zsign

    # elif key == ord('q'): R = TILT_LEFT @ R
    # elif key == ord('e'): R = TILT_LEFT.T @ R

    elif key == 81: rot[0] -= np.pi/90 #R = R @ LOOK_LEFT # LEFT-ARROW
    elif key == 82: rot[1] -= np.pi/90 #R = LOOK_UP @ R # UP-ARROW
    elif key == 83: rot[0] += np.pi/90 #R = R @ LOOK_LEFT.T # RIGHT-ARROW
    elif key == 84: rot[1] += np.pi/90 #R = LOOK_UP.T @ R # DOWN-ARROW

    elif key == ord('='): glob_scale *= 1.5 # +
    elif key == ord('-'): glob_scale /= 1.5 # -

    elif key == ord(']'): SPEED *= 1.5 # +
    elif key == ord('['): SPEED /= 1.5 # -

    elif key == 80: # HOME
        R = np.eye(3)
        loc = np.zeros(3)

    elif key > -1: print("Unkown key:", key)

cv2.destroyAllWindows()
