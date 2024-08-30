#!/bin/bash

set -ue

epochs=20
# epochs=1
# epochs=20
# epochs=100
# epochs=500

# Start timer
start_time=$(date +%s)

while true; do
    nvidia-smi --query-gpu=utilization.gpu --format=csv >> gpu_logs.txt;
    sleep 5;
done &

SPLIT=0

# Within-study
# All train outputs are saved in params["model_outdir"]
# SOURCE=CCLE
SOURCE=gCSI
TARGET=$SOURCE
echo "SOURCE: $SOURCE"
echo "TARGET: $TARGET"
echo "SPLIT:  $SPLIT"
python dualgcn_train_improve.py \
    --train_ml_data_dir ml_data/${SOURCE}-${TARGET}/split_${SPLIT} \
    --val_ml_data_dir ml_data/${SOURCE}-${TARGET}/split_${SPLIT} \
    --model_outdir out_model/${SOURCE}/split_${SPLIT} \
    --epochs $epochs \
    # --drug_path ml_data/#{SOURCE}-${TARGET}/split_${SPLIT}/drug_features \
    # --omics_path ml_data/#{SOURCE}-${TARGET}/split_${SPLIT}_omics_data \
    # --ppi_path ml_data/#{SOURCE}-${TARGET}/split_${SPLIT}_PPI/PPI_network_new.txt\
    # --cuda_name cuda:7

echo "GPU UTILIZATION"
awk 'NR%2==0 {print}' gpu_logs.txt | tr -d " %" > gpu_log_strip.txt
awk '{s+=$1}END{print "average %:",s/NR}' gpu_log_strip.txt
awk '{if($1>0+max){max=$1}} END{print "peak %:",max}' gpu_log_strip.txt

# # Cross-study
# # All train outputs are saved in params["model_outdir"]
# SOURCE=GDSCv1
# TARGET=CCLE
# echo "SOURCE: $SOURCE"
# echo "TARGET: $TARGET"
# echo "SPLIT:  $SPLIT"
# python graphdrp_train_improve.py \
#     --train_ml_data_dir ml_data/${SOURCE}-${TARGET}/split_${SPLIT} \
#     --val_ml_data_dir ml_data/${SOURCE}-${TARGET}/split_${SPLIT} \
#     --model_outdir out_model/${SOURCE}/split_${SPLIT} \
#     --cuda_name cuda:7
#     # --epochs $epochs \

# End timer
end_time=$(date +%s)

# Calculate elapsed time
elapsed_time=$((end_time - start_time))
echo "Elapsed time: $elapsed_time seconds"