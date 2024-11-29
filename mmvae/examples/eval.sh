# REAL_DATASET_DIR=/remote-home1/dataset/OpenMMLab___Kinetics-400/raw/Kinetics-400/videos_val/
EXP_NAME=release-panda70m
SAMPLE_RATE=1
NUM_FRAMES=33
RESOLUTION=256
METRIC=lpips

SUBSET_SIZE=0
REAL_DATASET_DIR=video_gen/${EXP_NAME}_sr${SAMPLE_RATE}_nf${NUM_FRAMES}_res${RESOLUTION}_subset${SUBSET_SIZE}/origin
echo $REAL_DATASET_DIR
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

echo $METRIC

if [[ $METRIC != "ssim" ]]; then
accelerate launch \
    --config_file examples/accelerate_configs/default_config.yaml \
    scripts/eval.py \
    --batch_size 1 \
    --real_video_dir ${REAL_DATASET_DIR} \
    --generated_video_dir video_gen/${EXP_NAME}_sr${SAMPLE_RATE}_nf${NUM_FRAMES}_res${RESOLUTION}_subset${SUBSET_SIZE} \
    --device cuda \
    --sample_fps 1 \
    --sample_rate ${SAMPLE_RATE} \
    --num_frames ${NUM_FRAMES} \
    --resolution ${RESOLUTION} \
    --crop_size ${RESOLUTION} \
    --subset_size ${SUBSET_SIZE} \
    --metric ${METRIC}
else
python scripts/eval.py \
    --mp \
    --batch_size 1 \
    --real_video_dir ${REAL_DATASET_DIR} \
    --generated_video_dir video_gen/${EXP_NAME}_sr${SAMPLE_RATE}_nf${NUM_FRAMES}_res${RESOLUTION}_subset${SUBSET_SIZE} \
    --device cpu \
    --sample_fps 1 \
    --sample_rate ${SAMPLE_RATE} \
    --num_frames ${NUM_FRAMES} \
    --resolution ${RESOLUTION} \
    --crop_size ${RESOLUTION} \
    --subset_size ${SUBSET_SIZE} \
    --metric ${METRIC}
fi