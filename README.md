
# Prepare env
```
conda create -n univa python=3.10 -y
conda activate univa

pip install --upgrade pip  # enable PEP 660 support
pip install -e .
pip install -e ".[train]"
pip install flash-attn --no-build-isolation

cd evaluation/VLMEvalKit
pip install -r requirements.txt
cd ../../

```

# Dataset

## Pretrain

**LLaVA-558k**

```
wget https://huggingface.co/datasets/LanguageBind/MoE-LLaVA/resolve/main/llava_image.zip
unzip llava_image.zip
```

**ShareGPT4V-1246K**

```
for i in $(seq -w 001 014); do
  wget "https://huggingface.co/datasets/LanguageBind/Video-LLaVA-1.5/resolve/main/sharegpt4v_2.zip.$i"
done
cat sharegpt4v_2.zip.0* > sharegpt4v_2.zip
unzip sharegpt4v_2.zip
unzip sharegpt4v.zip
```

**ALLaVA-707K**

```
TODO
```

## Finetune

**LLaVA-665K**

```
wget https://huggingface.co/datasets/LanguageBind/MoE-LLaVA/resolve/main/llava_image_tune_2.zip.001
wget https://huggingface.co/datasets/LanguageBind/MoE-LLaVA/resolve/main/llava_image_tune_2.zip.002
cat llava_image_tune_2.zip.00* > llava_image_tune_2.zip
unzip llava_image_tune_2.zip
unzip llava_image_tune.zip
```

**Cambrian-737K**

```
base_url="https://huggingface.co/datasets/LanguageBind/Cambrian737k/resolve/main/Cambrian737k/"
files=(ai2d chartqa coco docvqa dvqa gqa ocr_vqa textvqa vg)

for file in "${files[@]}"; do
  wget "${base_url}${file}.tar"
done

for file in "${files[@]}"; do
  tar -xvf ${file}.tar
done
```

**SMR-473K**

```
TODO
```

# Train

## Pretrain
```
cd /mnt/data/lb/UniVA/FlowWorld
conda activate univa
bash scripts/train_ross/pretrain_clip_qwen2_0p5.sh
```

## Finetune
```
cd /mnt/data/lb/UniVA/FlowWorld
conda activate univa
bash scripts/train_ross/finetune_clip_qwen2_0p5_chatml.sh
```

# Evaluation

## VLMEvalKit

The evaluation on [POPE](https://github.com/AoiDragon/POPE), [HallusionBench](https://github.com/tianyi-lab/HallusionBench), [MMBench_DEV_EN](https://github.com/open-compass/mmbench/), [MMBench_DEV_CN](https://github.com/open-compass/mmbench/), [SEEDBench_IMG](https://github.com/AILab-CVC/SEED-Bench), [MMMU_DEV_VAL](https://mmmu-benchmark.github.io/), [AI2D_TEST](https://allenai.org/data/diagrams), [OCRBench](https://github.com/Yuliang-Liu/MultimodalOCR), [RealWorldQA](https://x.ai/news/grok-1.5v), [DocVQA_VAL](https://www.docvqa.org/), [InfoVQA_VAL](https://www.docvqa.org/datasets/infographicvqa), [TextVQA_VAL](https://textvqa.org/), [GQA_TestDev_Balanced](https://cs.stanford.edu/people/dorarad/gqa/about.html), [](), [](), [](), [](), [](), [](), []() are implemented based on [VLMEvalKit](https://github.com/open-compass/VLMEvalKit).

The data of benchmarks will be downloaded automatically.

**Run VLMEvalKit**

```
conda activate univa
cd /mnt/data/lb/UniVA/FlowWorld/evaluation/VLMEvalKit


export LMUData="/mnt/data/lb/datasets/vlmevalkit_cache"
MODEL_PATH="/mnt/data/lb/logs/univa/univa-siglip-qwen2p5-3p0b-pt558k-sft737k-mmtag"
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 CUDA_LAUNCH_BLOCKING=1 torchrun --nproc-per-node=8 run.py \
    --data POPE HallusionBench MMBench_DEV_EN MMBench_DEV_CN SEEDBench_IMG MMMU_DEV_VAL AI2D_TEST OCRBench RealWorldQA \
    --model ${MODEL_PATH} \
    --judge exact_matching \
    --work-dir ${MODEL_PATH}/eval/VLMEvalKit

MODEL_PATH="/mnt/data/lb/logs/univa/univa-siglip-qwen2p5-3p0b-pt558k-sft737k-mmtag"
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 CUDA_LAUNCH_BLOCKING=1 torchrun --nproc-per-node=8 run.py \
    --data ChartQA_TEST DocVQA_VAL InfoVQA_VAL TextVQA_VAL GQA_TestDev_Balanced \
    --model ${MODEL_PATH} \
    --judge exact_matching \
    --work-dir ${MODEL_PATH}/eval/VLMEvalKit

```

**Print results quickly**

```
cd /mnt/data/lb/UniVA/FlowWorld
conda activate univa
vlmevalkit="/mnt/data/lb/logs/univa/univa-siglip-qwen2p5-3p0b-pt558k-sft737k-mmtag/eval/VLMEvalKit/T20250403_G1f803128"
python tools/print_results.py --vlmevalkit_root ${vlmevalkit} 
```

## MMVP

The evaluation on [MMVP](https://openaccess.thecvf.com/content/CVPR2024/papers/Tong_Eyes_Wide_Shut_Exploring_the_Visual_Shortcomings_of_Multimodal_LLMs_CVPR_2024_paper.pdf) is implemented based on [Cambrian-1](https://github.com/cambrian-mllm/cambrian/tree/main/eval/eval/mmvp).

**Prepare dataset (Optional)**

We prepare the offline MMVP data, download and unpackage to $HF_HOME.

```
wget https://huggingface.co/datasets/LanguageBind/Cambrian737k/resolve/main/mmvp_cache/mmvp_cache.tar
tar -xf mmvp_cache.tar  
# we will get a folder named "mmvp_cache"
```

**Run MMVP**

This step will download dataset if do not exist offline MMVP data.

```
conda activate univa
cd /mnt/data/lb/UniVA/FlowWorld/evaluation/MMVP

HF_HOME="/mnt/data/lb/datasets/mmvp_cache"
MODEL_PATH="/mnt/data/lb/logs/univa/univa-siglip-qwen2p5-3p0b-pt558k-sft737k-mmtag"
CONV_MODE="qwen_chatml"
HF_HOME=${HF_HOME} CUDA_VISIBLE_DEVICES=1 python mmvp_eval.py \
    --model_path ${MODEL_PATH} \
    --conv_mode ${CONV_MODE} \
    --answers_file ${MODEL_PATH}/eval/MMVP/answers.jsonl
```

**Print formatting results**

```
python mmvp_test.py \
    --answers_file ${MODEL_PATH}/eval/MMVP/answers.jsonl \
    --csv_file ${MODEL_PATH}/eval/MMVP/all_results.csv
```
