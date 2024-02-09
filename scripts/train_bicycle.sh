#!/bin/sh
cd ~/novel-viewpoint-synthesis/
python3 train.py \
    -s ~/data1/bicycle \
    --rescale 4 \
    --device cuda:1 \
    --sh-degree 0
