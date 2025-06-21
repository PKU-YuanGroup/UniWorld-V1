#!/bin/bash
export WANDB_MODE="online"
export WANDB_API_KEY="953e958793b218efb850fa194e85843e2c3bd88b"

export TOKENIZERS_PARALLELISM=true

export NCCL_IB_TC=136
export NCCL_IB_SL=5
export NCCL_IB_GID_INDEX=3
export NCCL_SOCKET_IFNAME=eth
export NCCL_IB_HCA=mlx5
export NCCL_IB_TIMEOUT=22
export NCCL_IB_QPS_PER_CONNECTION=8
export NCCL_NET_PLUGIN=none

# export ACCL_C4_STATS_MODE=CONN
# export ACCL_IB_SPLIT_DATA_NUM=4
# export ACCL_IB_QPS_LOAD_BALANCE=1
# export ACCL_IB_GID_INDEX_FIX=1
# export ACCL_LOG_TIME=1

MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
MASTER_PORT=${MASTER_PORT:-29500}
RANK=${RANK:-0}
WORLD_SIZE=${WORLD_SIZE:-1}
NUM_PROCESSES=$((8 * WORLD_SIZE))
# NUM_PROCESSES=1
# export  CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7
# NEED MODIFY in YAML:
  # data_txt
  # pretrained_lvlm_name_or_path
  # pretrained_denoiser_name_or_path
  # pretrained_mlp2_path
  # pretrained_siglip_mlp_path

accelerate launch \
  --config_file scripts/accelerate_configs/ddp_config.yaml \
  --main_process_ip ${MASTER_ADDR} \
  --main_process_port ${MASTER_PORT} \
  --machine_rank ${RANK} \
  --num_machines ${WORLD_SIZE} \
  --num_processes ${NUM_PROCESSES} \
  train_redux_ema.py \
  scripts/redux/flux_redux_1024_zerores_stage2_any15_lora.yaml