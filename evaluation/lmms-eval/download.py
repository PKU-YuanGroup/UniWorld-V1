from datasets import load_dataset

# 列出所有要下载的数据集
datasets_to_download = [
    "lmms-lab/POPE",
    "lmms-lab/HallusionBench",
    "lmms-lab/SEED-Bench",
    "lmms-lab/MMMU",
    "lmms-lab/ai2d",
]

for dataset_name in datasets_to_download:
    try:
        print(f"正在下载 {dataset_name}...")
        dataset = load_dataset(dataset_name)
        print(f"成功下载 {dataset_name}")
    except Exception as e:
        print(f"下载 {dataset_name} 时出错: {e}")

dataset = load_dataset("lmms-lab/MMBench", "cn")
dataset = load_dataset("lmms-lab/MMBench", "en")
