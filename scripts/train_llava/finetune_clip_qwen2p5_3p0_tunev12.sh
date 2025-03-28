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
JSON_FOLDER="/storage/lb/dataset/Cambrian737k"
IMAGE_FOLDER="/storage/lb/dataset/Cambrian737k"
LLM="/storage/lb/checkpoints/Qwen/Qwen2.5-3B-Instruct"
VISION_ENCODER="/storage/lb/logs/ross/llava-clip-qwen2p5-3p0b-pt558k-newenv-tunev/vision_tower"
PRETRAIN_DIR="/storage/lb/logs/ross/llava-clip-qwen2p5-3p0b-pt558k-newenv-tunev"
OUTPUT_DIR="/storage/lb/logs/ross/llava-clip-qwen2p5-3p0b-pt558k-sft737k-newenv-tunev12"
RUN_NAME="llava-clip-qwen2p5-3p0b-pt558k-sft737k-newenv-tunev12"

mkdir -p ${OUTPUT_DIR}

torchrun --nproc-per-node=8 --nnodes 1 --node_rank 0 \
    --master_addr="localhost" --master_port="29805" \
    \
    train.py \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --learning_rate 2e-5 \
    --warmup_ratio 0.03 \
    \
    --unfreeze_mm_vision_tower True \
    --mm_vision_tower_lr 2e-6 \
    \
    --deepspeed ./scripts/zero2.json \
    --model_name_or_path ${LLM} \
    --pretrain_mm_mlp_adapter ${PRETRAIN_DIR}/mm_projector.bin \
    --output_dir ${OUTPUT_DIR} \
    --vision_tower ${VISION_ENCODER} \
    --version qwen_chatml \
    \
    --data_path ${JSON_FOLDER}/Cambrian737k.json \
    --image_folder ${IMAGE_FOLDER} \
    \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --image_aspect_ratio pad \
    --group_by_modality_length True \
    --bf16 True \
    --num_train_epochs 1 \
    --per_device_eval_batch_size 4 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 24000 \
    --save_total_limit 1 \
    --save_only_model \
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
