
RESULT_DIR="/data/logs/tad/train_feature"
mkdir -p ${RESULT_DIR}


DATA_PATH="/data/checkpoints/LanguageBind/offline_feature/offline_dit_s_feature_256/imagenet_val_256"
cd /data/FlowWorld
conda activate dit


keys=("t" "c" "x")
for i in "${!keys[@]}"; do
    key=${keys[$i]}

    RESULT_FILE_PATH="${RESULT_DIR}/${key}.txt"
    if [ -f "$RESULT_FILE_PATH" ]; then
        echo "File $RESULT_FILE_PATH exists. Skipping..."
        continue
    fi

    CUDA_VISIBLE_DEVICES=0 python tools/train_feature.py \
        --data_path ${DATA_PATH} \
        --data_key ${key} >> ${RESULT_FILE_PATH}
    sleep 1s
done



for i in {0..11}
do 

    RESULT_FILE_PATH="${RESULT_DIR}/feature_${i}.txt"
    if [ -f "$RESULT_FILE_PATH" ]; then
        echo "File $RESULT_FILE_PATH exists. Skipping..."
        continue
    fi

    CUDA_VISIBLE_DEVICES=0 python tools/train_feature.py \
        --data_path ${DATA_PATH} \
        --data_key "feature" \
        --feature_layer_idx ${i} >> ${RESULT_FILE_PATH}
    sleep 1s
done