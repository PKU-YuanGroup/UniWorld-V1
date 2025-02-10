



# extract_feature
torchrun --nproc_per_node 8 --master_port 29502 -m extract_feature \
    --data_path /mnt/data/lb/ImageNet-1K/train \
    --data_split imagenet_train \
    --output_path /mnt/data/lb/offline_vae_512 \
    --vae /mnt/data/lb/checkpoint/stabilityai/sd-vae-ft-ema \
    --image_size 512 \
    --batch_size 20 \
    --num_workers 16 






tmux new -d -s clash "cd /storage/clash && mkdir -p ~/.config/clash && cp Country.mmdb ~/.config/clash && ./clash -f 1730517458971.yml"

conda activate dit_lb
cd /storage/lb/LightningDiT
accelerate launch \
    --main_process_ip 127.0.0.1 \
    --main_process_port 1235 \
    --machine_rank 0 \
    --num_processes 8 \
    --num_machines 1 \
    train.py \
    --config configs/baseline___.yaml




conda activate dit_lb
cd /storage/lb/LightningDiT
accelerate launch \
    --main_process_ip 127.0.0.1 \
    --main_process_port 1235 \
    --machine_rank 0 \
    --num_processes 8 \
    --num_machines 1 \
    inference.py \
    --config configs/xt_8_learn.yaml \
    --demo


conda activate llamageneval
python tools/evaluator.py \
    /storage/lb/datasets/VIRTUAL_imagenet256_labeled.npz \
    ../logs/fastdit/100kx1024_lrx2/lightningdit-xl-2-ckpt-0100000-250.npz







accelerate launch \
    --main_process_ip 127.0.0.1 \
    --main_process_port 1235 \
    --machine_rank 0 \
    --num_processes 8 \
    --num_machines 1 \
    inference.py \
    --config configs/xt_4_learn.yaml 


accelerate launch \
    --main_process_ip 127.0.0.1 \
    --main_process_port 1235 \
    --machine_rank 0 \
    --num_processes 8 \
    --num_machines 1 \
    inference.py \
    --config configs/xt_12.yaml 


accelerate launch \
    --main_process_ip 127.0.0.1 \
    --main_process_port 1235 \
    --machine_rank 0 \
    --num_processes 8 \
    --num_machines 1 \
    inference.py \
    --config configs/xt_12_learn.yaml 


accelerate launch \
    --main_process_ip 127.0.0.1 \
    --main_process_port 1235 \
    --machine_rank 0 \
    --num_processes 8 \
    --num_machines 1 \
    inference.py \
    --config configs/xt_16.yaml 


accelerate launch \
    --main_process_ip 127.0.0.1 \
    --main_process_port 1235 \
    --machine_rank 0 \
    --num_processes 8 \
    --num_machines 1 \
    inference.py \
    --config configs/xt_20_learn.yaml 


accelerate launch \
    --main_process_ip 127.0.0.1 \
    --main_process_port 1235 \
    --machine_rank 0 \
    --num_processes 8 \
    --num_machines 1 \
    inference.py \
    --config configs/xt_24.yaml 


