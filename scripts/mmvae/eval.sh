
REAL_DATASET_DIR=/storage/dataset/val2017
RESOLUTION=256
CKPT=/storage/lb/FlowWorld/results/debug_vae-lr1.00e-05-bs16-rs256-i1-t1-clip1-lpips1.0

accelerate launch \
    --config_file configs/accelerate_configs/default_config.yaml \
    mmvae/scripts/eval.py \
    --batch_size 1 \
    --image_path ${REAL_DATASET_DIR} \
    --device cuda \
    --resolution ${RESOLUTION} \
    --num_workers 8 \
    --from_pretrained ${CKPT} \
    --model_name MMVAE 
    # \
    # --hf_model "/storage/lb/DiT/cache_dir/models--stabilityai--sd-vae-ft-ema/snapshots/f04b2c4b98319346dad8c65879f680b1997b204a"