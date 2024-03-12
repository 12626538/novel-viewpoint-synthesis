#!/bin/sh
if [ -d "/home/jip/novel-viewpoint-synthesis/" ]
then
    CODE_DIR="/home/jip/novel-viewpoint-synthesis/";
    DATA_DIR="/home/jip/data1/";
elif [ -d "/media/jip/T7/thesis/code/novel-viewpoint-synthesis/" ]
then
    CODE_DIR="/media/jip/T7/thesis/code/novel-viewpoint-synthesis/";
    DATA_DIR="/media/jip/T7/thesis/code/data/";
fi

python3 $CODE_DIR/train.py \
    -s "$DATA_DIR/bonsai" \
    --rescale 2 \
    --device cuda:1 \
    --test-at 7000 11000 14000 17000 20000 25000 30000 \
    --save-at 7000 30000 \
    --sh-degree 3 \
    --grad-threshold 0.0005 \
    --densify-from 500 \
    --densify-every 100 \
    --reset-opacity-every 3000 \
    --loss-weight-mae 0.8 \
    --loss-weight-dssim 0.2 \
    --min-opacity 0.1 \
    --max-screen-size 0.15 \
    --max-density 0.05
