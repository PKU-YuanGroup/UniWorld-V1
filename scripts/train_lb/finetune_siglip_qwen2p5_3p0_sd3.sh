#!/bin/bash
export WANDB_API_KEY="953e958793b218efb850fa194e85843e2c3bd88b"
# NCCL setting
export NCCL_IB_TC=136
export NCCL_IB_SL=5
export NCCL_IB_GID_INDEX=3
export NCCL_SOCKET_IFNAME=eth
export NCCL_IB_HCA=mlx5
export NCCL_IB_TIMEOUT=22
export NCCL_IB_QPS_PER_CONNECTION=8
export NCCL_NET_PLUGIN=none


cd /mnt/data/lb/UniVA/FlowWorld
JSON_FOLDER="/mnt/data/datasets/train_json"
IMAGE_FOLDER="/mnt/data/datasets/LLaVA"
LLM="/mnt/data/checkpoints/Qwen/Qwen2.5-3B-Instruct"
VISION_ENCODER="/mnt/data/checkpoints/google/siglip-so400m-patch14-384"
DENOISE_DECODER="/mnt/data/checkpoints/stabilityai/stable-diffusion-3-medium-diffusers"
PRETRAIN_DIR="/mnt/data/lb/logs/univa/univa-siglip-qwen2p5-3p0b-pt558k-sft737k-mmtag-0403-stage3-pt"
OUTPUT_DIR="/mnt/data/lb/logs/univa/univa-siglip-qwen2p5-3p0b-pt558k-sft737k-mmtag-0403-stage3-ft"
RUN_NAME="univa-siglip-qwen2p5-3p0b-pt558k-sft737k-mmtag-0403-stage3-ft"

UNFREEZE_DENOISE_TOWER=False
mkdir -p ${OUTPUT_DIR}

torchrun --nproc-per-node=8 --nnodes 1 --node_rank 0 \
    --master_addr="localhost" --master_port="29805" \
    \
    train.py \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 1 \
    --learning_rate 2e-4 \
    --warmup_ratio 0.03 \
    \
    --denoise_tower ${DENOISE_DECODER} \
    --mm_denoise_projector_type mlp2x_gelu \
    --unfreeze_mm_denoise_tower ${UNFREEZE_DENOISE_TOWER} \
    \
    --mm_use_im_start_end \
    \
    --deepspeed ./scripts/zero2.json \
    --model_name_or_path ${LLM} \
    --output_dir ${OUTPUT_DIR} \
    --vision_tower ${VISION_ENCODER} \
    --version qwen_chatml \
    \
    --data_path ${JSON_FOLDER}/univa_tune__.json ${JSON_FOLDER}/univa_tune__.json ${JSON_FOLDER}/univa_tune__.json ${JSON_FOLDER}/univa_tune__.json \
    --image_folder ${IMAGE_FOLDER} \
    \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --image_aspect_ratio pad \
    --group_by_modality_length True \
    --bf16 True \
    --num_train_epochs 100 \
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
    --report_to tensorboard \
    --run_name ${RUN_NAME} > ${OUTPUT_DIR}/log.txt