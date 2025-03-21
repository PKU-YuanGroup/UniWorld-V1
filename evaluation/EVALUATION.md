# Evaluating Ross

## Overview
This directory contains the evaluation scripts and benchmarks for Ross. 
It includes a wide range of benchmarks to assess the model's performance across various tasks and domains.

## Evaluation using VLMEvalKit
We utilized [VLMEvalKit](https://github.com/open-compass/VLMEvalKit) to evaluate Ross on:
1. POPE
2. HallusionBench
3. MMBench (English and Chinese)
4. SEED
5. MMMU
6. AI2D
7. OCRBench
8. RealWorldQA


### Installation

```bash
cd VLMEvalKit
pip install -r requirements.txt
```

### Evaluation

```bash
CUDA_LAUNCH_BLOCKING=1 torchrun --nproc-per-node=8 run.py \
    --data POPE HallusionBench MMBench_DEV_EN MMBench_DEV_CN SEEDBench_IMG MMMU_DEV_VAL AI2D_TEST OCRBench RealWorldQA \
    --model ross-qwen2-7b, ross-vicuna-13b \
    --judge exact_matching
```


## Evaluation using lmms-eval
We utilized [lmms-eval](https://github.com/EvolvingLMMs-Lab/lmms-eval) to evaluate Ross on:
1. ChartQA
2. DocVQA
3. InfoVQA
4. TextVQA
5. GQA
6. MMLU
7. HellaSwag
8. IFEval

### Installation

```bash
cd lmms-eval
pip install -e .
```

### Evaluation

```bash
python -m accelerate.commands.launch \
    --num_processes=8 \
    -m lmms_eval \
    --model ross \
    --model_args "pretrained=HaochenWang/ross-qwen2-7b,conv_template=qwen_2,device_map=auto" \
    --tasks chartqa,docvqa_val,infovqa_val,textvqa_val,gqa,mmlu,hellaswag,ifeval \
    --batch_size 1 \
    --log_samples \
    --output_path ./results/ross-qwen2-7b
```

## Evaluation on MMVP
The evaluation on [MMVP](https://openaccess.thecvf.com/content/CVPR2024/papers/Tong_Eyes_Wide_Shut_Exploring_the_Visual_Shortcomings_of_Multimodal_LLMs_CVPR_2024_paper.pdf) is implemented based on [Cambrian-1](https://github.com/cambrian-mllm/cambrian/tree/main/eval/eval/mmvp).

```bash
cd MMVP

python mmvp_eval.py \
    --model_path HaochenWang/ross-qwen2-7b \
    --conv_mode qwen_2 \
    --answers_file ./answers/ross-qwen2-7b.jsonl

python mmvp_test.py \
    --answers_file ./answers/ross-qwen2-7b.jsonl \
    --csv_file ./all_results.csv
```
