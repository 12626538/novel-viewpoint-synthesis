#!/bin/sh
if [ -d "/home/jip/novel-viewpoint-synthesis" ]
then
    CODE_DIR="/home/jip/novel-viewpoint-synthesis";
    SOURCE_DIR="/home/jip/data1";
elif [ -d "/media/jip/T7/thesis/code/novel-viewpoint-synthesis" ]
then
    CODE_DIR="/media/jip/T7/thesis/code/novel-viewpoint-synthesis";
    SOURCE_DIR="/media/jip/T7/thesis/code/data";
fi

python3 $CODE_DIR/train.py \
    -s "$SOURCE_DIR/3du_data_7" \
    --device cuda:0 \
    --iterations 30000 \
    --test-at 5000 7000 11000 14000 17000 20000 25000 30000 \
    --save-at 7000 15000 22000 30000 \
    --densify-until -1 \
    --reset-opacity-until -1 \
    --lr-scales 0.005 \
    --lr-opacities 0.05 \
    --lr-colors 0.005 \
    --oneup-sh-every 2000 \
    --loss-weight-mae 0.0 \
    --loss-weight-mse 0.0 \
    --loss-weight-dssim 0.0 \
    --loss-weight-lpips 1.0 \
    --model-name "3du_data_7_deblur3" \
    --no-pbar
