# prepare env

```
conda create -n dit python=3.10 -y
pip install -e .
```

# extract feature

```
cd /data/FlowWorld
conda activate dit
torchrun --nproc_per_node 8 --master_port 29502 -m tools.extract_features \
    --data_path /data/OpenDataLab___ImageNet-1K/raw/ImageNet-1K/train \
    --data_split imagenet_train \
    --output_path /data/checkpoints/LanguageBind/offline_feature/offline_sdvae_256_path \
    --vae_type sdvae \
    --vae_path /data/checkpoints/stabilityai/sd-vae-ft-ema/vae-ft-ema-560000-ema-pruned.safetensors \
    --image_size 256 \
    --batch_size 50 \
    --num_workers 16 
```


```
cd /data/FlowWorld
conda activate dit
torchrun --nproc_per_node 8 --master_port 29502 -m tools.extract_features \
    --data_path /data/OpenDataLab___ImageNet-1K/raw/ImageNet-1K/train \
    --data_split imagenet_train \
    --output_path /data/checkpoints/LanguageBind/offline_feature/offline_vavae_256_path \
    --vae_type vavae \
    --vae_path /data/checkpoints/hustvl/vavae-imagenet256-f16d32-dinov2/vavae-imagenet256-f16d32-dinov2.pt \
    --image_size 256 \
    --batch_size 50 \
    --num_workers 16 
```


# train


## diffusion model
### single node

```
cd /data/FlowWorld
conda activate dit
accelerate launch \
    --main_process_ip 127.0.0.1 \
    --main_process_port 1236 \
    --machine_rank 0 \
    --num_processes 8 \
    --num_machines 1 \
    train.py \
    --config configs/flow_b_1000kx1024_sdvae.yaml
```

### multi node

```
cd /data/FlowWorld
conda activate dit
accelerate launch \
    --config_file configs/accelerate_configs/multi_node_example_by_ddp.yaml \
    --machine_rank 0 \
    train.py \
    --config configs/flow_s_1000kx1024_sdvae.yaml

cd /data/FlowWorld
conda activate dit
accelerate launch \
    --config_file configs/accelerate_configs/multi_node_example_by_ddp.yaml \
    --machine_rank 1 \
    train.py \
    --config configs/flow_s_1000kx1024_sdvae.yaml
```

## tokenizer

```
cd /data/FlowWorld
conda activate dit
accelerate launch \
    --main_process_ip 127.0.0.1 \
    --main_process_port 1236 \
    --machine_rank 0 \
    --num_processes 8 \
    --num_machines 1 \
    train_flowvae.py \
    --config configs/flowsdvae_50kx512_lgnm1p0.yaml
```

```
cd /data/FlowWorld
conda activate dit
accelerate launch \
    --config_file configs/accelerate_configs/multi_node_example_by_ddp.yaml \
    --machine_rank 0 \
    train_flowvae.py \
    --config configs/flowsdvae_500kx512_lgn0p0.yaml

cd /data/FlowWorld
conda activate dit
accelerate launch \
    --config_file configs/accelerate_configs/multi_node_example_by_ddp.yaml \
    --machine_rank 1 \
    train_flowvae.py \
    --config configs/flowsdvae_500kx512_lgn0p0.yaml

cd /data/FlowWorld
conda activate dit
accelerate launch \
    --config_file configs/accelerate_configs/multi_node_example_by_ddp.yaml \
    --machine_rank 2 \
    train_flowvae.py \
    --config configs/flowsdvae_500kx512_lgn0p0.yaml

cd /data/FlowWorld
conda activate dit
accelerate launch \
    --config_file configs/accelerate_configs/multi_node_example_by_ddp.yaml \
    --machine_rank 3 \
    train_flowvae.py \
    --config configs/flowsdvae_500kx512_lgn0p0.yaml
```

# inference

## demo

```
cd /data/FlowWorld
conda activate dit
accelerate launch \
    --main_process_ip 127.0.0.1 \
    --main_process_port 1234 \
    --machine_rank 0 \
    --num_processes 8 \
    --num_machines 1 \
    inference.py \
    --config configs/flow_s_1000kx1024_sdvae.yaml \
    --demo 
```

### flowvae
```
cd /data/FlowWorld
conda activate dit
accelerate launch \
    --main_process_ip 127.0.0.1 \
    --main_process_port 1234 \
    --machine_rank 0 \
    --num_processes 8 \
    --num_machines 1 \
    tools/evaluate_flowvae.py \
    --config configs/flowsdvae_50kx512_l1.yaml \
    --demo 
```

## batch inference


```
cd /data/FlowWorld
conda activate dit
accelerate launch \
    --main_process_ip 127.0.0.1 \
    --main_process_port 1235 \
    --machine_rank 0 \
    --num_processes 8 \
    --num_machines 1 \
    inference.py \
    --config configs/flow_s_1000kx1024_sdvae.yaml
```

```
cd /data/FlowWorld
conda activate dit
accelerate launch \
    --main_process_ip 127.0.0.1 \
    --main_process_port 1234 \
    --machine_rank 0 \
    --num_processes 8 \
    --num_machines 1 \
    tools/evaluate_flowvae.py \
    --config configs/flowsdvae_50kx512_lgnm1p0.yaml
```

# eval model

## prepare env

```
conda create -n dit_eval python=3.10 -y
conda activate dit_eval
# cuda12.2
pip install tensorflow==2.15.0 scipy requests tqdm numpy==1.23.5
pip install nvidia-pyindex
pip install nvidia-cublas-cu12 nvidia-cuda-cupti-cu12 nvidia-cuda-runtime-cu12 nvidia-cudnn-cu12

```

## fid
```
cd /data/FlowWorld
conda activate dit_eval
python tools/evaluator.py \
    /data/checkpoints/VIRTUAL_imagenet256_labeled.npz \
    /data/logs/tpt/diff_s_1000kx1024_fp32/dit-s-2-ckpt-1000000-250-diffusion.npz

```


# eval tokenizer
```
cd /data/FlowWorld
conda activate dit
accelerate launch \
    --main_process_ip 127.0.0.1 \
    --main_process_port 1234 \
    --machine_rank 0 \
    --num_processes 8 \
    --num_machines 1 \
    tools/evaluate_tokenizer.py \
    --ckpt_path /data/checkpoints/hustvl/vavae-imagenet256-f16d32-dinov2/vavae-imagenet256-f16d32-dinov2.pt \
    --model_type vavae \
    --data_path /data/OpenDataLab___ImageNet-1K/raw/ImageNet-1K/val \
    --output_path /data/logs/flow/vavae
    
```

```
cd /data/FlowWorld
conda activate dit
accelerate launch \
    --main_process_ip 127.0.0.1 \
    --main_process_port 1234 \
    --machine_rank 0 \
    --num_processes 8 \
    --num_machines 1 \
    tools/evaluate_tokenizer.py \
    --ckpt_path /data/checkpoints/stabilityai/sd-vae-ft-ema/vae-ft-ema-560000-ema-pruned.safetensors \
    --model_type sdvae \
    --data_path /data/OpenDataLab___ImageNet-1K/raw/ImageNet-1K/val \
    --output_path /data/logs/flow/sdvae
    
```