# from tqdm import tqdm
# import shutil
# import os
# import json

# json_path = '/mnt/data/datasets/Cambrian737k.json'
# image_root = '/storage/nyw/Cambrian-10M'
# save_root = '/mnt/data/datasets/Cambrian737k'
# with open(json_path, 'r', encoding='utf-8') as f:
#     data = json.load(f)
# image_paths = [os.path.join(image_root, i['image']) for i in tqdm(data) if 'image' in i]
# print('total:', len(image_paths))
# image_paths = list(set(image_paths))
# print('unique:', len(image_paths))
# # for i in tqdm(image_paths):
# #     assert os.path.exists(i)
# # print('all images exist')

# os.makedirs(save_root, exist_ok=True)
# for i in tqdm(image_paths):
#     relative_path = os.path.relpath(i, image_root)
#     dst = os.path.join(save_root, relative_path)
#     dir_path = os.path.dirname(dst)
#     os.makedirs(dir_path, exist_ok=True)
#     src = i
#     shutil.copy(src, dst)
# print('done')

import os
import json
import shutil
from tqdm import tqdm
from multiprocessing import Pool, cpu_count

json_path = '/mnt/data/datasets/Cambrian737k.json'
image_root = '/storage/nyw/Cambrian-10M'
save_root = '/mnt/data/datasets/Cambrian737k'

with open(json_path, 'r', encoding='utf-8') as f:
    data = json.load(f)

image_paths = list(set(os.path.join(image_root, i['image']) for i in data if 'image' in i))
print('Total unique images:', len(image_paths))

os.makedirs(save_root, exist_ok=True)

def copy_file(src):
    relative_path = os.path.relpath(src, image_root)
    dst = os.path.join(save_root, relative_path)
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    shutil.copy(src, dst)

# 使用多进程加速复制
with Pool(processes=128) as pool:
    list(tqdm(pool.imap_unordered(copy_file, image_paths), total=len(image_paths)))

print('Copying completed.')