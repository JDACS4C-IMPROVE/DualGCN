set -ue

# ----------------------------------------
# CSA (cross-study analysis) exmple
# ----------------------------------------
# export IMPROVE_DATA_DIR="./csa_data/"
# export PYTHONPATH=$PYTHONPATH:/homes/ac.cesarasa/Project_DualGCN/IMPROVE

SPLIT=0
epochs=200

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
    --ml_data_outdir ml_data/${SOURCE}-${TARGET}/split_${SPLIT}

python dualgcn_train_improve.py \
    --train_ml_data_dir ml_data/${SOURCE}-${TARGET}/split_${SPLIT} \
    --val_ml_data_dir ml_data/${SOURCE}-${TARGET}/split_${SPLIT} \
    --model_outdir out_model/${SOURCE}/split_${SPLIT} \
    --epochs $epochs \

python dualgcn_infer_improve.py \
    --test_ml_data_dir ml_data/${SOURCE}-${TARGET}/split_${SPLIT} \
    --model_dir out_model/${SOURCE}/split_${SPLIT} \
    --infer_outdir out_infer/${SOURCE}-${TARGET}/split_${SPLIT} \

echo 'Finished!'