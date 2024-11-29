# conda activate fl
# 953e958793b218efb850fa194e85843e2c3bd88b
cd /storage/lb/DiT
accelerate launch \
    --config_file accelerate_configs/deepspeed_zero2_config.yaml \
    train.py \
    --ema_deepspeed_config_file accelerate_configs/zero3.json \
    --data_path /storage/dataset/imagenet/imagenet/train \
    --num_workers 16 \
    --model FlowWorld-XL/2 \
    --output_dir="debug" \
    --proj_name "debug" \
    --log_name "debug" \
    --resume_from_checkpoint "latest" \
    --use_ema \
    --ema_update_freq 1 \
    --ema_decay 0.9999 \
    --train_batch_size 256 \
    --checkpointing_steps 1000 \
    --gradient_checkpointing