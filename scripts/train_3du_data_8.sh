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
    -s "$SOURCE_DIR/3du_data_8" \
    --load-checkpoint "$SOURCE_DIR/3du_data_8/" \
    --device cuda:0 \
    --iterations 10000 \
    --test-at 5000 7000 11000 14000 17000 20000 25000 30000 \
    --save-at 7000 15000 22000 30000 \
    --densify-until -1 \
    --reset-opacity-until -1 \
    --oneup-sh-every 2000 \
    --lr-quats 0.1 \
    --lr-scales 0.05 \
    --lr-opacities 0.05 \
    --lr-positions 0.0 \
    --lr-colors 0.05 \
    --loss-weight-mae 0.0 \
    --loss-weight-mse 0.4 \
    --loss-weight-dssim 0.2 \
    --loss-weight-lpips 0.1
    # --do-blur
    # --no-pbar \
