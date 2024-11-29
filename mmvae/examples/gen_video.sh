export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
REAL_DATASET_DIR=/storage/dataset/vae_eval/panda70m
EXP_NAME=release-panda70m
SAMPLE_RATE=1
NUM_FRAMES=33
RESOLUTION=256
CKPT=/storage/lcm/WF-VAE_paper/backup/results/wfvae-16dim-L-release
SUBSET_SIZE=0

accelerate launch \
    --config_file examples/accelerate_configs/default_config.yaml \
    scripts/recon_video.py \
    --batch_size 4 \
    --real_video_dir ${REAL_DATASET_DIR} \
    --generated_video_dir video_gen/${EXP_NAME}_sr${SAMPLE_RATE}_nf${NUM_FRAMES}_res${RESOLUTION}_subset${SUBSET_SIZE} \
    --device cuda \
    --sample_fps 24 \
    --sample_rate ${SAMPLE_RATE} \
    --num_frames ${NUM_FRAMES} \
    --resolution ${RESOLUTION} \
    --subset_size ${SUBSET_SIZE} \
    --num_workers 8 \
    --from_pretrained ${CKPT} \
    --model_name WFVAE \
    --output_origin \
    --crop_size ${RESOLUTION}
