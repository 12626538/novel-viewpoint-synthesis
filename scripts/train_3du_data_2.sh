#!/bin/sh
cd ~/novel-viewpoint-synthesis/
python3 train.py \
    -s ~/data1/3du_data_2 \
    --rescale 1 \
    --device cuda:1 \
    --num-points 30000 \
    --densify-every 300 \
    --grad-threshold 1e-4
