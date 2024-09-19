#!/bin/bash

set -ue

# ----------------------------------------
# CSA (cross-study analysis) exmple
# ----------------------------------------
# export IMPROVE_DATA_DIR="./csa_data/"
# export PYTHONPATH=$PYTHONPATH:/homes/ac.cesarasa/Project_DualGCN/IMPROVE
SPLIT=0

# Within-study
SOURCE=gCSI
TARGET=$SOURCE
echo "SOURCE: $SOURCE"
echo "TARGET: $TARGET"
echo "SPLIT:  $SPLIT"

python dualgcn_preprocess_improve.py \
    --train_split_file ${SOURCE}_split_${SPLIT}_train.txt \
    --val_split_file ${SOURCE}_split_${SPLIT}_val.txt \
    --test_split_file ${TARGET}_split_${SPLIT}_test.txt \
    --output_dir ml_data/${SOURCE}-${TARGET}/split_${SPLIT}
