import argparse
import os
import numpy as np
import cv2

import torch
from torchvision.utils import save_image

from src.model.gaussians import Gaussians,RenderPackage
from src.data import DataSet
from src.arg import ModelParams,DataParams,PipeLineParams,get_args
from src.camera import Camera
from scipy.spatial.transform import Rotation

import time

mouse_x,mouse_y = 0,0
def cb_mouse(event, x, y, flags, param):
    global mouse_x,mouse_y

    mouse_x,mouse_y=x,y

    if event == cv2.EVENT_LBUTTONDOWN:
        print("CLICK",(x,y))
        split_frac = x / W
    elif event == cv2.EVENT_LBUTTONUP:
        pass


parser = argparse.ArgumentParser("Training Novel Viewpoint Synthesis")
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

# Use checkpoint
if not os.path.exists(data_args.source_path):
    print(f"Dataset {data_args.source_path} not found")
    exit(1)
# 3DU datasets are from intrinsics and extrinsics
if "3du" in data_args.source_path:
    dataset = DataSet.from_3du(device=args.device, **vars(data_args))
else:
    dataset = DataSet.from_colmap(device=args.device, **vars(data_args))

split_frac=.5
glob_scale = 1
bg = torch.tensor([0.,0.,0.],dtype=torch.float32, device=args.device)


cv2.namedWindow("image")
cv2.setMouseCallback("image", cb_mouse)

cam_idx = 0
while True:
    camera = dataset.cameras[ cam_idx%len(dataset) ]
    render_pkg = model.render(camera, glob_scale=glob_scale, bg=bg)

    render = (render_pkg.rendering.permute(1,2,0).detach().cpu().numpy()*255).astype(np.uint8)
    gt_image = (camera.gt_image.permute(1,2,0).detach().cpu().numpy()*255).astype(np.uint8)

    # RGB -> BGR
    render = render[:,:,::-1]
    gt_image = gt_image[:,:,::-1]

    W = render.shape[1]
    W_split = round(W * split_frac)
    render[:,:W_split] = gt_image[:,:W_split]

    # cv2.putText(render, f"Camera: {camera.name}", [10,40], cv2.FONT_HERSHEY_PLAIN, .7, (0,0,0), 1, cv2.LINE_AA)
    cv2.imshow('image',render)
    key = cv2.waitKey(0)


    if key == 27: break # ESC

    elif key == 81: cam_idx -= 1 # LEFT-ARROW
    elif key == 83: cam_idx += 1 #R = R @ LOOK_LEFT.T # RIGHT-ARROW

    elif key == ord('='): glob_scale *= 1.5 # +
    elif key == ord('-'): glob_scale /= 1.5 # -

    elif key == ord(' '): split_frac = mouse_x / W

    elif key == 80: # HOME
        cam_idx = 0

    elif key > -1: print("Unkown key:", key)

cv2.destroyAllWindows()
