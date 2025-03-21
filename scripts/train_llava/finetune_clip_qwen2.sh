#!/bin/bash



torchrun --nproc-per-node=8 --nnodes 1 --node_rank 0 \
    --master_addr="localhost" --master_port="29805" \
    \
    train.py \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --learning_rate 2e-5 \
    --warmup_ratio 0.03 \
    \
    --deepspeed ./scripts/zero2.json \
    --model_name_or_path /data/llm_model_zoo/Qwen2-7B-Instruct \
    --pretrain_mm_mlp_adapter ./checkpoints/llava-v1.5-clip-qwen2-7b-pt558k/mm_projector.bin \
    --output_dir ./checkpoints/llava-v1.5-clip-qwen2-7b-pt558k-sft737k \
    --vision_tower /data/llm_model_zoo/clip-vit-large-patch14-336 \
    --version qwen_2 \
    \
    --data_path ./playground/data/cambrian737k.json \
    --image_folder ./playground/data \
    \
    --mm_projector_type mlp2x_gelu \
    --mm_inv_projector_type denoiser_vit3x \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
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
    --model_max_length 32768 \
    --gradient_checkpointing True \
    --dataloader_num_workers 16 \
    --lazy_preprocess True \
    --report_to wandb \
    --run_name "llava-v1.5-clip-qwen2-7b-pt558k-sft737k"
