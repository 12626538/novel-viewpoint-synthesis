#!/bin/sh

if [ "$#" -ne 1 ]; then
    echo "Usage: train_3du_data.sh DATASET"
    exit 2
fi

if [ -d "/home/jip/novel-viewpoint-synthesis" ]; then
    CODE_DIR="/home/jip/novel-viewpoint-synthesis";
    SOURCE_DIR="/home/jip/data1/3du_data/$1";
elif [ -d "/media/jip/T7/thesis/code/novel-viewpoint-synthesis" ]; then
    CODE_DIR="/media/jip/T7/thesis/code/novel-viewpoint-synthesis";
    SOURCE_DIR="/media/jip/T7/thesis/code/data/3du_data/$1";
else
    echo "Could find resolve code and data directories"
    exit 2
fi

if [ ! -d $SOURCE_DIR ]; then
    echo "Could not find source dir $SOURCE_DIR"
    exit 2
fi

python3 $CODE_DIR/train.py \
    -s "$SOURCE_DIR" \
    --rescale 1 \
    --device cuda:1 \
    --test-at 5000 7000 11000 14000 17000 20000 25000 30000 \
    --save-at 7000 15000 22000 30000 \
    --densify-from 500 \
    --densify-every 100 \
    --reset-opacity-every 3000 \
    --grad-threshold 0.0002 \
    --max-density 0.05 \
    --min-opacity 0.1 \
    --max-screen-size 0.15 \
    --loss-weight-mse 1.0 \
    --loss-weight-dssim 0.2 \
    --loss-weight-lpips 0.1 \
    --loss-weight-mae 0.8 \
    --block-size 14
