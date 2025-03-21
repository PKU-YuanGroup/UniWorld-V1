#!/bin/bash



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
    --model_name_or_path Qwen/Qwen2-7B-Instruct \
    --output_dir ./checkpoints/llava-v1.5-clip-qwen2-7b-pt558k \
    --vision_tower openai/clip-vit-large-patch14-336 \
    --version plain \
    \
    --data_path ./playground/data/LLaVA-Pretrain/blip_laion_cc_sbu_558k.json \
    --image_folder ./playground/data/LLaVA-Pretrain/images \
    \
    --mm_projector_type mlp2x_gelu \
    --tune_mm_mlp_adapter True \
    --mm_inv_projector_type denoiser_vit3x \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
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
    --model_max_length 8192 \
    --gradient_checkpointing True \
    --dataloader_num_workers 16 \
    --lazy_preprocess True \
    --report_to wandb \
    --run_name "llava-v1.5-siglip-qwen2-7b-pt558k"
