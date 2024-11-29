export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
REAL_DATASET_DIR=/storage/dataset/val2017
RESOLUTION=256
CKPT=/storage/lb/DiT/WF-VAE/results/debug_vae-lr1.00e-05-bs8-rs256-sr1-fr1

accelerate launch \
    --config_file examples/accelerate_configs/default_config.yaml \
    scripts/eval_image_lzj.py \
    --batch_size 1 \
    --image_path ${REAL_DATASET_DIR} \
    --device cuda \
    --resolution ${RESOLUTION} \
    --num_workers 8 \
    --from_pretrained ${CKPT} \
    --model_name MMVAE \
    --output_save_dir tmp_save_eval
