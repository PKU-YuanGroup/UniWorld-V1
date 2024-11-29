

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nnodes=1 --nproc_per_node 8 --master_port 29513 \
    -m sample \
    --model_path debug/checkpoint-88000/model \
    --num_frames 1 \
    --height 256 \
    --width 256 \
    --label label.txt \
    --save_img_path "test_save_88k_" \
    --guidance_scale 4.0 \
    --num_sampling_steps 250 \
    --seed 1234 \
    --num_samples_per_label 8