import argparse
import glob
import multiprocessing as mp
import os

# fmt: off
import sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))
# fmt: on

import tempfile
import time
import warnings

import cv2
import numpy as np
import tqdm

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import Compose, ToTensor, Normalize

from detectron2.config import get_cfg
from detectron2.data.detection_utils import read_image
from detectron2.projects.deeplab import add_deeplab_config
from detectron2.utils.logger import setup_logger
from detectron2.modeling import build_model
from detectron2.data import MetadataCatalog
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.engine.defaults import DefaultPredictor


from .mask2former.config import add_maskformer2_config
from pathlib import Path
import skimage.io


class SegmentationModel:
    def __init__(self, config_file, model_weights):
        # load config from file and command-line arguments
        cfg = get_cfg()
        add_deeplab_config(cfg)
        add_maskformer2_config(cfg)
        cfg.merge_from_file(config_file)
        cfg.merge_from_list(['MODEL.WEIGHTS', model_weights])
        cfg.freeze()

        self.metadata = MetadataCatalog.get(
            cfg.DATASETS.TEST[0] if len(cfg.DATASETS.TEST) else "__unused"
        )
        self.predictor = DefaultPredictor(cfg)

        self.CLASSES = self.metadata.get("stuff_classes")
        self.PALETTE = self.metadata.get("stuff_colors")

    def segment(self, input_imgs, output_file=None):
        predictions = self.predictor(input_imgs)
        color_seg_list = []

        for prediction, input_img in zip(predictions, input_imgs):
            seg_file = (prediction["sem_seg"].argmax(dim=0)).cpu().numpy().astype(np.uint8)

            palette = np.array(self.PALETTE)
            assert palette.shape[0] == len(self.CLASSES)
            assert palette.shape[1] == 3
            assert len(palette.shape) == 2
            color_seg = np.zeros((seg_file.shape[0], seg_file.shape[1], 3), dtype=np.uint8)
            for label, color in enumerate(palette):
                color_seg[seg_file == label, :] = color

            img_seg_ = input_img * 0.5 + color_seg * 0.5
            img_seg = img_seg_.astype(np.uint8)

            combine = np.hstack((input_img, img_seg_, color_seg))

            if output_file:
                if not os.path.isdir(output_file):
                    os.makedirs(output_file)

                seg_name = "segmentation"
                out_file_img_seg = "color"

                skimage.io.imsave(+"jpg", combine.astype(np.uint8))
                skimage.io.imsave(str(out_file_img_seg)[:-3] + "png", color_seg)
                skimage.io.imsave(str(seg_name)[:-3] + "png", seg_file + 1)

            color_seg_list.append(color_seg)

        return color_seg_list

class SegmentationModelBatched:
    def __init__(self, config_file, model_weights):
        # load config from file and command-line arguments
        cfg = get_cfg()
        add_deeplab_config(cfg)
        add_maskformer2_config(cfg)
        cfg.merge_from_file(config_file)
        cfg.merge_from_list(['MODEL.WEIGHTS', model_weights])
        cfg.freeze()

        self.metadata = MetadataCatalog.get(
            cfg.DATASETS.TEST[0] if len(cfg.DATASETS.TEST) else "__unused"
        )

        checkpointer = DetectionCheckpointer(self.model)
        checkpointer.load(cfg.MODEL.WEIGHTS)

        self.CLASSES = self.metadata.get("stuff_classes")
        self.PALETTE = self.metadata.get("stuff_colors")

        self.model = build_model(cfg)
        self.model.eval()
