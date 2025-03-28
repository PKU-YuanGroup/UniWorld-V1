#!/bin/bash
export WANDB_API_KEY="953e958793b218efb850fa194e85843e2c3bd88b"
# NCCL setting
export GLOO_SOCKET_IFNAME=bond0
export NCCL_SOCKET_IFNAME=bond0
export NCCL_IB_HCA=mlx5_10:1,mlx5_11:1,mlx5_12:1,mlx5_13:1
export NCCL_IB_GID_INDEX=3
export NCCL_IB_TC=162
export NCCL_IB_TIMEOUT=25
export NCCL_PXN_DISABLE=0
export NCCL_IB_QPS_PER_CONNECTION=4
export NCCL_ALGO=Ring
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NCCL_IB_RETRY_CNT=32
export TOKENIZERS_PARALLELISM=false


cd /storage/lb/ross
conda activate ross_env
JSON_FOLDER="/storage/lb/dataset/LanguageBind/MoE-LLaVA/train_json"
IMAGE_FOLDER="/storage/lb/dataset/LanguageBind/MoE-LLaVA"
LLM="/storage/lb/checkpoints/Qwen/Qwen2.5-0.5B-Instruct"
VISION_ENCODER="/storage/lb/checkpoints/openai/clip-vit-large-patch14-336"
OUTPUT_DIR="/storage/lb/logs/ross/llava-clip-qwen2p5-0p5b-pt558k-newenv"
RUN_NAME="llava-clip-qwen2p5-0p5b-pt558k-newenv"

mkdir -p ${OUTPUT_DIR}

torchrun --nproc-per-node=8 --nnodes 1 --node_rank 0 \
    --master_addr="localhost" --master_port="29805" \
    \
    train.py \
    --per_device_train_batch_size 16 \
    --gradient_accumulation_steps 2 \
    --learning_rate 1e-3 \
    --warmup_ratio 0.03 \
    \
    --deepspeed ./scripts/zero2.json \
    --model_name_or_path ${LLM} \
    --output_dir ${OUTPUT_DIR} \
    --vision_tower ${VISION_ENCODER} \
    --version plain \
    \
    --data_path ${JSON_FOLDER}/llava_image_.json \
    --image_folder ${IMAGE_FOLDER} \
    \
    --mm_projector_type mlp2x_gelu \
    --tune_mm_mlp_adapter True \
    --mm_vision_select_layer -2 \
    --bf16 True \
    --num_train_epochs 1 \
    --per_device_eval_batch_size 4 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 24000 \
    --save_total_limit 1 \
    --weight_decay 0. \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 4096 \
    --gradient_checkpointing True \
    --dataloader_num_workers 16 \
    --lazy_preprocess True \
    --report_to wandb \
    --run_name ${RUN_NAME} >> ${OUTPUT_DIR}/log.txt
