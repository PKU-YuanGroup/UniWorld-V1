import os
import json
import random
from tqdm import tqdm
from pathlib import Path

with open('tools/imagenet1k_class_to_word.json', 'r') as f:
    synset_to_name = json.load(f)

def generate_llava_data(imagenet_root):
    data = []
    imagenet_root = Path(imagenet_root)
    
    for class_id in tqdm(os.listdir(imagenet_root)):
        class_dir = imagenet_root / class_id
        if not class_dir.is_dir() or class_id not in synset_to_name:
            continue
        
        label_name = synset_to_name[class_id]
        for img_path in class_dir.glob("*.JPEG"):
            img_rel_path = img_path.relative_to(imagenet_root).as_posix()
            sample = {
                "id": img_path.stem,
                "image": img_rel_path,
                "conversations": [
                    {"from": "human", "value": "Render a clear and concise summary of the photo.\n<image>"},
                    {"from": "gpt", "value": label_name}
                ]
            }
            data.append(sample)
            
    random.shuffle(data)
    with open(f'imagenet1k_llava_num{len(data)}.json', "w") as f:
        json.dump(data, f, indent=2)

if __name__ == '__main__':
    imagenet1k_path = '/storage/dataset/OpenDataLab___ImageNet-1K/raw/ImageNet-1K/train'
    generate_llava_data(imagenet1k_path)