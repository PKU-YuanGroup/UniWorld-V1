tmux new -d -s clash "cd /data/clash && mkdir -p ~/.config/clash && cp Country.mmdb ~/.config/clash && ./clash -f 1724121750205.yml"

conda create -n dit python=3.10 -y
pip install -e .

# extract_feature
cd /data/FlowWorld
conda activate dit
torchrun --nproc_per_node 8 --master_port 29502 -m tools.extract_feature \
    --data_path /mnt/data/lb/ImageNet-1K/train \
    --data_split imagenet_train \
    --output_path /mnt/data/lb/offline_vae_512 \
    --vae /mnt/data/lb/checkpoint/stabilityai/sd-vae-ft-ema \
    --image_size 512 \
    --batch_size 20 \
    --num_workers 16 

# train
cd /data/FlowWorld
conda activate dit
accelerate launch \
    --main_process_ip 127.0.0.1 \
    --main_process_port 1235 \
    --machine_rank 0 \
    --num_processes 8 \
    --num_machines 1 \
    train.py \
    --config configs/did_s_100kx1024_qf4x4_img1p0.yaml

conda activate dit
accelerate launch \
    --main_process_ip 127.0.0.1 \
    --main_process_port 1234 \
    --machine_rank 0 \
    --num_processes 8 \
    --num_machines 1 \
    train.py \
    --config configs/ft/did_s_100kx1024_qf4x4_img1p0_train_all.yaml

# inference
cd /data/FlowWorld
conda activate dit
accelerate launch \
    --main_process_ip 127.0.0.1 \
    --main_process_port 1235 \
    --machine_rank 0 \
    --num_processes 8 \
    --num_machines 1 \
    inference.py \
    --config configs/ft/did_s_100kx1024_qf1x1_img0p0_uncond_zeroqf_train_all.yaml \
    --demo 

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
    /data/logs/moc/ft_did_s_100kx1024_qf1x1_img1p0_train_qf_em/did-s-2-ckpt-0100000-250-diffusion.npz