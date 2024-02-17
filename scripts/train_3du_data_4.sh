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
    -s "$DATA_DIR/3du_data_4" \
    --device cuda:1 \
    --load-checkpoint "$DATA_DIR/3du_data_4/pointcloud.ply" \
    --model-name "3du_data_4_nodensify_nocolors" \
    --test-at 5000 7000 11000 14000 17000 20000 25000 30000 \
    --save-at 7000 22000 30000 \
    --sh-degree 0 \
    --max-screen-size 50 \
    --densify-until -1 \
    --max-density 0.001
