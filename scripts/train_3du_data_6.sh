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
    -s "$SOURCE_DIR/3du_data_6" \
    --device cuda:0 \
    --iterations 20000 \
    --test-at 5000 7000 11000 14000 17000 20000 25000 30000 \
    --save-at 30000 \
    --densify-until -1 \
    --reset-opacity-until -1 \
    --lr-scales 0.005 \
    --lr-opacities 0.05 \
    --lr-colors 0.005 \
    --oneup-sh-every 3000 \
    --model-name "3du_data_6_lpips"

python3 $CODE_DIR/train.py \
    -s "$SOURCE_DIR/3du_data_6" \
    --load-checkpoint $CODE_DIR/models/3du_data_6_lpips/iter_20000/point_cloud.ply
