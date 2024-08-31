#!/bin/bash

set -ue

SPLIT=0

# Within-study
SOURCE=gCSI
TARGET=$SOURCE
echo "SOURCE: $SOURCE"
echo "TARGET: $TARGET"
echo "SPLIT:  $SPLIT"
python dualgcn_infer_improve.py \
    --test_ml_data_dir ml_data/${SOURCE}-${TARGET}/split_${SPLIT} \
    --model_dir out_model/${SOURCE}/split_${SPLIT} \
    --infer_outdir out_infer/${SOURCE}-${TARGET}/split_${SPLIT} \
    # --cuda_name cuda:7