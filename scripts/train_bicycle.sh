#!/bin/sh
if [ -d "~/novel-viewpoint-synthesis/" ]
then
    CODE_DIR="~/novel-viewpoint-synthesis/";
    DATA_DIR="~/data1/";
elif [ -d "/media/jip/T7/thesis/code/novel-viewpoint-synthesis/" ]
then
    CODE_DIR="/media/jip/T7/thesis/code/novel-viewpoint-synthesis/";
    DATA_DIR="/media/jip/T7/thesis/code/data/";
fi

python3 $CODE_DIR/train.py \
    -s "$DATA_DIR/bicycle" \
    --rescale 4 \
    --device cuda:0 \
    --sh-degree 0 \
    --test-at 5000 7000 11000 14000 17000 20000 25000 30000 \
    --save-at 7000 14000 20000 25000 30000 \
