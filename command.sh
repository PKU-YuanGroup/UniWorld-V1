tmux new -d -s clash "cd /data/clash && mkdir -p ~/.config/clash && cp Country.mmdb ~/.config/clash && ./clash -f 1724121750205.yml"

conda create -n dit python=3.10 -y
pip install -e .

# extract_feature
cd /data/FlowWorld
conda activate dit
torchrun --nproc_per_node 8 --master_port 29502 -m tools.extract_features \
    --data_path /data/OpenDataLab___ImageNet-1K/raw/ImageNet-1K/train \
    --data_split imagenet_train \
    --output_path /data/checkpoints/LanguageBind/offline_feature/offline_vae_256_path \
    --vae /data/checkpoints/stabilityai/sd-vae-ft-ema \
    --image_size 256 \
    --batch_size 50 \
    --num_workers 16 


cd /data/FlowWorld
conda activate dit
torchrun --nproc_per_node 8 --master_port 29502 -m tools.extract_dit_features \
    --data_path /data/OpenDataLab___ImageNet-1K/raw/ImageNet-1K/val \
    --data_split imagenet_val \
    --output_path /data/checkpoints/LanguageBind/offline_feature/offline_dit_s_feature_256 \
    --vae /data/checkpoints/stabilityai/sd-vae-ft-ema \
    --image_size 256 \
    --batch_size 1 \
    --num_workers 16 \
    --num_diffusion_steps 1000 \
    --config configs/diff_s_1000kx1024_fp32_get_feature.yaml

# train
cd /data/FlowWorld
conda activate dit
accelerate launch \
    --main_process_ip 127.0.0.1 \
    --main_process_port 1236 \
    --machine_rank 0 \
    --num_processes 8 \
    --num_machines 1 \
    train.py \
    --config configs/diff_s_1000kx1024_fp32.yaml

    
cd /data/FlowWorld
conda activate dit
accelerate launch \
    --main_process_ip 127.0.0.1 \
    --main_process_port 1235 \
    --machine_rank 0 \
    --num_processes 8 \
    --num_machines 1 \
    train.py \
    --config configs/diff_128_s_p1_100kx1024.yaml

cd /data/FlowWorld
conda activate dit
accelerate launch \
    --config_file configs/accelerate_configs/multi_node_example_by_ddp.yaml \
    --machine_rank 0 \
    train.py \
    --config configs/diff_l_1000kx1024_fp32_4node.yaml

cd /data/FlowWorld
conda activate dit
accelerate launch \
    --config_file configs/accelerate_configs/multi_node_example_by_ddp.yaml \
    --machine_rank 1 \
    train.py \
    --config configs/diff_l_1000kx1024_fp32_4node.yaml

cd /data/FlowWorld
conda activate dit
accelerate launch \
    --config_file configs/accelerate_configs/multi_node_example_by_ddp.yaml \
    --machine_rank 2 \
    train.py \
    --config configs/diff_l_1000kx1024_fp32_4node.yaml

cd /data/FlowWorld
conda activate dit
accelerate launch \
    --config_file configs/accelerate_configs/multi_node_example_by_ddp.yaml \
    --machine_rank 3 \
    train.py \
    --config configs/diff_l_1000kx1024_fp32_4node.yaml




cd /data/FlowWorld
conda activate dit
accelerate launch \
    --config_file configs/accelerate_configs/multi_node_example_by_ddp.yaml \
    --machine_rank 0 \
    train.py \
    --config configs/diff_b_1000kx1024_qknorm.yaml

cd /data/FlowWorld
conda activate dit
accelerate launch \
    --config_file configs/accelerate_configs/multi_node_example_by_ddp.yaml \
    --machine_rank 1 \
    train.py \
    --config configs/diff_b_1000kx1024_qknorm.yaml

cd /data/FlowWorld
conda activate dit
accelerate launch \
    --config_file configs/accelerate_configs/multi_node_example_by_ddp.yaml \
    --machine_rank 2 \
    train.py \
    --config configs/diff_b_1000kx1024_qknorm.yaml

cd /data/FlowWorld
conda activate dit
accelerate launch \
    --config_file configs/accelerate_configs/multi_node_example_by_ddp.yaml \
    --machine_rank 3 \
    train.py \
    --config configs/diff_b_1000kx1024_qknorm.yaml

# inference
cd /data/FlowWorld
conda activate dit
accelerate launch \
    --main_process_ip 127.0.0.1 \
    --main_process_port 1234 \
    --machine_rank 0 \
    --num_processes 8 \
    --num_machines 1 \
    inference.py \
    --config configs/diff_s_1000kx1024_qknorm.yaml \
    --demo 


cd /data/FlowWorld
conda activate dit
accelerate launch \
    --main_process_ip 127.0.0.1 \
    --main_process_port 1235 \
    --machine_rank 0 \
    --num_processes 8 \
    --num_machines 1 \
    inference.py \
    --config configs/diff_s_1000kx1024_qknorm.yaml


# inference
cd /data/FlowWorld
conda activate dit
accelerate launch \
    --main_process_ip 127.0.0.1 \
    --main_process_port 1235 \
    --machine_rank 0 \
    --num_processes 8 \
    --num_machines 1 \
    inference_did.py \
    --config configs/did_s_100kx1024_qf1x1_img1p0.yaml \
    --demo 

# evaluator
conda create -n dit_eval python=3.10 -y
conda activate dit_eval
# cuda12.2
pip install tensorflow==2.15.0 scipy requests tqdm numpy==1.23.5
pip install nvidia-pyindex
pip install nvidia-cublas-cu12 nvidia-cuda-cupti-cu12 nvidia-cuda-runtime-cu12 nvidia-cudnn-cu12

cd /data/FlowWorld
conda activate dit_eval
python tools/evaluator.py \
    /data/checkpoints/VIRTUAL_imagenet256_labeled.npz \
    /data/logs/tpt/diff_s_1000kx1024_fp32/dit-s-2-ckpt-1000000-250-diffusion.npz

cd /data/FlowWorld
conda activate dit_eval
python tools/evaluator.py \
    /data/checkpoints/VIRTUAL_imagenet128_labeled.npz \
    /data/logs/tpt/diff_b_400kx256/dit-b-2-ckpt-0200000-250-diffusion.npz