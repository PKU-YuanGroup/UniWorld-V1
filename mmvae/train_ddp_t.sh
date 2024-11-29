export WANDB_MODE="offline"
export WANDB_PROJECT=debug_vae
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export GLOO_SOCKET_IFNAME=bond0
export NCCL_SOCKET_IFNAME=bond0
export NCCL_IB_HCA=mlx5_10:1,mlx5_11:1,mlx5_12:1,mlx5_13:1
export NCCL_IB_GID_INDEX=3
export NCCL_IB_TC=162
export NCCL_IB_TIMEOUT=22
export NCCL_PXN_DISABLE=0
export NCCL_IB_QPS_PER_CONNECTION=4
export NCCL_ALGO=Ring
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
# export NCCL_DEBUG=DETAIL
# export TORCH_DISTRIBUTED_DEBUG=DETAIL
# conda activate fl
# 953e958793b218efb850fa194e85843e2c3bd88b

EXP_NAME=debug_vae
torchrun \
    --nnodes=1 --nproc_per_node=8 \
    --master_addr=localhost \
    --master_port=12135 \
    train_ddp.py \
    --exp_name ${EXP_NAME} \
    --video_path /storage/dataset/imagenet/imagenet/train \
    --eval_video_path /storage/dataset/val2017/ \
    --model_name MMVAE \
    --model_config /storage/lb/DiT/WF-VAE/config_t.json \
    --resolution 256 \
    --num_frames 1 \
    --batch_size 8 \
    --lr 0.00001 \
    --epochs 4 \
    --disc_start 50000 \
    --dataset_num_worker 16 \
    --save_ckpt_step 5000 \
    --eval_steps 1000 \
    --eval_batch_size 1 \
    --eval_num_frames 33 \
    --eval_sample_rate 1 \
    --eval_subset_size 5000 \
    --eval_lpips \
    --ema \
    --ema_decay 0.999 \
    --perceptual_weight 0.1 \
    --loss_type l1 \
    --sample_rate 1 \
    --disc_cls causalvideovae.model.losses.LPIPSWithDiscriminator2D \
    --wavelet_loss \
    --wavelet_weight 0.1 \
    --train_text 