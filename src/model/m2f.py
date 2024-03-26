import numpy as np

from detectron2.config import get_cfg
from detectron2.projects.deeplab import add_deeplab_config
from detectron2.engine.defaults import DefaultPredictor


from .mask2former.config import add_maskformer2_config


class SegmentationModel:
    CLASSES = [
        'wall', 'floor', 'ceiling', 'bed', 'window', 'cabinet', 'door', 'table (excluding coffee-table, side-table, sofa-table, desk), end-table, occasional-table',
        'plant', 'curtain, drape, valance', 'chair (excluding accent-chair, armchair, recliner)', 'painting, poster, photo', 'sofa, couch', 'mirror', 'rug, carpet, mat',
        'accent-chair, armchair, recliner', 'desk', 'wardrobe, hall-tree, closet', 'table-lamp, floor-lamp', 'bathtub',
        'throw-pillow, decorative-pillow, pillow, cuschion, floor-pillow', 'boxes, basket', 'dresser, chest', 'counter, countertop, kitchen-island', 'sink', 'fireplace',
        'fridge, refrigerator', 'stair', 'bookshelf, decorative-ledge, bookcase', 'window-blind, window-shutter', 'coffe-table, side-table, sofa-table', 'toilet', 'book',
        'kitchen-stove', 'computer, laptop, notebook', 'swivel-chair', 'towel', 'overhead-lighting, chandelier, pendant, pendent', 'tv, tv-screen, screen, monitor, television',
        'cloth', 'fence, bannister, balauster, handrail', 'ottoman, footstool', 'bottle', 'washer, washing-machine, dryer, drying-machine', 'game, puzzle, toy', 'bag',
        'oven, furnace', 'microwave, microwave-oven', 'flowerpot', 'bicycle', 'dishwasher', 'blanket, throw, sofa-blanket', 'kitchen-air-extractor, hood, exhaust-hood',
        'sconce, wall-sconce', 'bin', 'fan', 'shower', 'radiator', 'wall-clock, father-clock', 'window-frame', 'door-frame',
        'decor-accent, table-clock, candle, candleholder, lantern, bookend', 'wall-decoration, art', 'curtain-rod',
        'sound-system, audio-system, speaker, loud-speaker, sound-box, sounding-box, stereo', 'piano', 'guitar', 'wall-switch', 'room-divider', 'telephone', 'fireplace-screen',
        'dog-bed, cat-bed', 'kitchen-utensil', 'crockery, dish', 'cutting-board', 'pan, kitchen-pot', 'magazine-rack, magazine-basket', 'coat-rack', 'fireplace-tool',
        'sport-machine', 'tripod', 'printer', 'wire', 'keyboard', 'mouse', 'pad', 'bed-frame', 'balcony', 'stuff', 'board', 'toilet-paper', 'heater', 'receiver', 'remote',
        'hanger', 'soap-dispenser', 'plug', 'flush-button', 'alarm', 'shoe-rack', 'shoe', 'hair-dryer', 'temperature-controller', 'pipe', 'charger', 'ironing-table', 'shower-head',
        'cage', 'hat', 'vacuum-cleaner', 'tent', 'drum, drum-stick', 'toilet-brush', 'baggage, luggage, suitcase', 'door-glass', 'tv-unit', 'water-pump', 'stand', 'storage', 'unknown',
    ]

    PALETTE = np.array(
        [[128, 0, 0], [0, 128, 0], [128, 128, 0], [0, 0, 128], [128, 0, 128], [0, 128, 128], [128, 128, 128], [64, 0, 0], [192, 0, 0], [64, 128, 0], [192, 128, 0], [64, 0, 128],
        [192, 0, 128], [64, 128, 128], [192, 128, 128], [0, 64, 0], [128, 64, 0], [0, 192, 0], [128, 192, 0], [0, 64, 128], [128, 64, 128], [0, 192, 128], [128, 192, 128],
        [64, 64, 0], [192, 64, 0], [64, 192, 0], [192, 192, 0], [64, 64, 128], [192, 64, 128], [64, 192, 128], [192, 192, 128], [0, 0, 64], [128, 0, 64], [0, 128, 64],
        [128, 128, 64], [0, 0, 192], [128, 0, 192], [0, 128, 192], [128, 128, 192], [64, 0, 64], [192, 0, 64], [64, 128, 64], [192, 128, 64], [64, 0, 192], [192, 0, 192],
        [64, 128, 192], [192, 128, 192], [0, 64, 64], [128, 64, 64], [0, 192, 64], [128, 192, 64], [0, 64, 192], [128, 64, 192], [0, 192, 192], [128, 192, 192], [64, 64, 64],
        [192, 64, 64], [64, 192, 64], [192, 192, 64], [64, 64, 192], [192, 64, 192], [64, 192, 192], [192, 192, 192], [32, 0, 0], [160, 0, 0], [32, 128, 0], [160, 128, 0],
        [32, 0, 128], [160, 0, 128], [32, 128, 128], [160, 128, 128], [96, 0, 0], [224, 0, 0], [96, 128, 0], [224, 128, 0], [96, 0, 128], [224, 0, 128], [96, 128, 128],
        [224, 128, 128], [32, 64, 0], [160, 64, 0], [32, 192, 0], [160, 192, 0], [32, 64, 128], [160, 64, 128], [32, 192, 128], [160, 192, 128], [96, 64, 0], [224, 64, 0],
        [96, 192, 0], [224, 192, 0], [96, 64, 128], [224, 64, 128], [96, 192, 128], [224, 192, 128], [32, 0, 64], [160, 0, 64], [32, 128, 64], [160, 128, 64], [32, 0, 192],
        [160, 0, 192], [32, 128, 192], [160, 128, 192], [96, 0, 64], [224, 0, 64], [96, 128, 64], [224, 128, 64], [96, 0, 192], [224, 0, 192], [96, 128, 192], [224, 128, 192],
        [32, 64, 64], [160, 64, 64], [32, 192, 64], [160, 192, 64], [32, 64, 192], [160, 64, 192], [32, 192, 192], [160, 192, 192], [96, 64, 64]],
        dtype=np.uint8
    )


    def __init__(self, config_file, model_weights):
        # load config from file and command-line arguments
        cfg = get_cfg()
        add_deeplab_config(cfg)
        add_maskformer2_config(cfg)
        cfg.merge_from_file(config_file)
        cfg.merge_from_list(['MODEL.WEIGHTS', model_weights])
        cfg.freeze()

        self.predictor = DefaultPredictor(cfg)

        import torch
        torch.uint8

    def segment(self, input_img:np.ndarray, alpha:float=.4) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Segment image

        Args:
        - `input_img:np.ndarray` of shape H,W,C and dtype np.unin8
        - `alpha:float=0.4` is the blending value for `combined` return value

        Returns:
        - `labels:np.ndarray` of shape H,W where each 'pixel' is the predicted
            class label
        - `color_seg:np.ndarray` of shape H,W,C. Same as labels, except using
            the color of each class as defined in `PALETTE`
        - `combined:np.ndarray` is `input_img` overlayed with `color_seg`
            with the provided `alpha` value for blending.
        """
        # Forward pass
        logits = self.predictor(input_img)['sem_seg']

        # Convert to class labels
        labels = (logits.argmax(dim=0)).cpu().numpy().astype(np.uint8)

        # Apply palette
        color_seg = self.PALETTE[labels]

        # Blend input with colors
        combined = ( alpha * color_seg + (1-alpha) * input_img ).astype(np.uint8)

        return labels, color_seg, combined
