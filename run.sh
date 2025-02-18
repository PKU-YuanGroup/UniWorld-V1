
cd /data/FlowWorld
conda activate dit
accelerate launch \
    --main_process_ip 127.0.0.1 \
    --main_process_port 1235 \
    --machine_rank 0 \
    --num_processes 8 \
    --num_machines 1 \
    train_tad.py \
    --config configs/tad_128_s_p1_100kx1024_cls0p1.yaml


cd /data/FlowWorld
conda activate dit
accelerate launch \
    --main_process_ip 127.0.0.1 \
    --main_process_port 1235 \
    --machine_rank 0 \
    --num_processes 8 \
    --num_machines 1 \
    train_tad.py \
    --config configs/tad_128_s_p1_100kx1024_cls0p01.yaml


cd /data/FlowWorld
conda activate dit
accelerate launch \
    --main_process_ip 127.0.0.1 \
    --main_process_port 1235 \
    --machine_rank 0 \
    --num_processes 8 \
    --num_machines 1 \
    train_tad.py \
    --config configs/tad_128_s_p1_100kx1024_tw.yaml











cd /data/FlowWorld
conda activate dit
accelerate launch \
    --main_process_ip 127.0.0.1 \
    --main_process_port 1235 \
    --machine_rank 0 \
    --num_processes 8 \
    --num_machines 1 \
    train.py \
    --config configs/pretrain/diff_128_s_p1_100kx1024.yaml


cd /data/FlowWorld
conda activate dit
accelerate launch \
    --main_process_ip 127.0.0.1 \
    --main_process_port 1235 \
    --machine_rank 0 \
    --num_processes 8 \
    --num_machines 1 \
    train.py \
    --config configs/finetune/diff_128_s_p1_100kx1024.yaml







cd /data/FlowWorld
conda activate dit
accelerate launch \
    --main_process_ip 127.0.0.1 \
    --main_process_port 1235 \
    --machine_rank 0 \
    --num_processes 8 \
    --num_machines 1 \
    train_tad.py \
    --config configs/pretrain/tad_128_s_p1_100kx1024_cls0p1.yaml


cd /data/FlowWorld
conda activate dit
accelerate launch \
    --main_process_ip 127.0.0.1 \
    --main_process_port 1235 \
    --machine_rank 0 \
    --num_processes 8 \
    --num_machines 1 \
    train_tad.py \
    --config configs/finetune/tad_128_s_p1_100kx1024_cls0p1.yaml




cd /data/FlowWorld
conda activate dit
accelerate launch \
    --main_process_ip 127.0.0.1 \
    --main_process_port 1235 \
    --machine_rank 0 \
    --num_processes 8 \
    --num_machines 1 \
    train_tad.py \
    --config configs/pretrain/tad_128_s_p1_100kx1024_cls0p01.yaml


cd /data/FlowWorld
conda activate dit
accelerate launch \
    --main_process_ip 127.0.0.1 \
    --main_process_port 1235 \
    --machine_rank 0 \
    --num_processes 8 \
    --num_machines 1 \
    train_tad.py \
    --config configs/finetune/tad_128_s_p1_100kx1024_cls0p01.yaml



cd /data/FlowWorld
conda activate dit
accelerate launch \
    --main_process_ip 127.0.0.1 \
    --main_process_port 1235 \
    --machine_rank 0 \
    --num_processes 8 \
    --num_machines 1 \
    train_tad.py \
    --config configs/pretrain/tad_128_s_p1_100kx1024_cls0p1_tw.yaml


cd /data/FlowWorld
conda activate dit
accelerate launch \
    --main_process_ip 127.0.0.1 \
    --main_process_port 1235 \
    --machine_rank 0 \
    --num_processes 8 \
    --num_machines 1 \
    train_tad.py \
    --config configs/finetune/tad_128_s_p1_100kx1024_cls0p1_tw.yaml


















cd /data/FlowWorld
conda activate dit
accelerate launch \
    --main_process_ip 127.0.0.1 \
    --main_process_port 1235 \
    --machine_rank 0 \
    --num_processes 8 \
    --num_machines 1 \
    train_tad.py \
    --config configs/tad_128_s_p1_100kx1024_cls0p1_mlp2at4.yaml


cd /data/FlowWorld
conda activate dit
accelerate launch \
    --main_process_ip 127.0.0.1 \
    --main_process_port 1235 \
    --machine_rank 0 \
    --num_processes 8 \
    --num_machines 1 \
    train_tad.py \
    --config configs/tad_128_s_p1_100kx1024_cls0p1_dec2at4.yaml


cd /data/FlowWorld
conda activate dit
accelerate launch \
    --main_process_ip 127.0.0.1 \
    --main_process_port 1235 \
    --machine_rank 0 \
    --num_processes 8 \
    --num_machines 1 \
    train_tad.py \
    --config configs/tad_128_s_p1_100kx1024_cls0p01_dec2.yaml


cd /data/FlowWorld
conda activate dit
accelerate launch \
    --main_process_ip 127.0.0.1 \
    --main_process_port 1235 \
    --machine_rank 0 \
    --num_processes 8 \
    --num_machines 1 \
    train_tad.py \
    --config configs/tad_128_s_p1_100kx1024_cls0p01_dec2at4.yaml


cd /data/FlowWorld
conda activate dit
accelerate launch \
    --main_process_ip 127.0.0.1 \
    --main_process_port 1235 \
    --machine_rank 0 \
    --num_processes 8 \
    --num_machines 1 \
    train_tad.py \
    --config configs/tad_128_s_p1_100kx1024_cls0p5_dec2at4.yaml


cd /data/FlowWorld
conda activate dit
accelerate launch \
    --main_process_ip 127.0.0.1 \
    --main_process_port 1235 \
    --machine_rank 0 \
    --num_processes 8 \
    --num_machines 1 \
    train_tad.py \
    --config configs/tad_128_s_p1_100kx1024_cls0p05_dec2at4.yaml


cd /data/FlowWorld
conda activate dit
accelerate launch \
    --main_process_ip 127.0.0.1 \
    --main_process_port 1235 \
    --machine_rank 0 \
    --num_processes 8 \
    --num_machines 1 \
    train_tad.py \
    --config configs/tad_128_s_p1_100kx1024_cls0p25_dec2at4.yaml




DATA_PATH="/data/checkpoints/LanguageBind/offline_feature/offline_dit_s_feature_256/imagenet_val_256"
cd /data/FlowWorld
conda activate dit
CUDA_VISIBLE_DEVICES=0 python tools/train_feature.py \
        --data_path ${DATA_PATH} \
        --data_key "x" \
        --feature_layer_idx 0


DATA_PATH="/data/checkpoints/LanguageBind/offline_feature/offline_dit_s_feature_256/imagenet_val_256"
cd /data/FlowWorld
conda activate dit
CUDA_VISIBLE_DEVICES=0 python tools/train_feature.py \
        --data_path ${DATA_PATH} \
        --data_key "feature" \
        --feature_layer_idx 0


DATA_PATH="/data/checkpoints/LanguageBind/offline_feature/offline_dit_s_feature_256/imagenet_val_256"
cd /data/FlowWorld
conda activate dit
CUDA_VISIBLE_DEVICES=0 python tools/log_reg.py \
        --data_path ${DATA_PATH} \
        --data_key "t"