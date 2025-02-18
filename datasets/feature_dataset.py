# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# LightningDiT: https://github.com/hustvl/LightningDiT
# --------------------------------------------------------

import os
import numpy as np
from glob import glob
from tqdm import tqdm
import torch
from torch.utils.data import Dataset
from safetensors import safe_open


class FeatureDataset(Dataset):
    def __init__(self, data_dir):
        self.data_dir = data_dir

        self.files = sorted(glob(os.path.join(data_dir, "*.safetensors")))
        self.img_to_file_map = self.get_img_to_safefile_map()

    def get_img_to_safefile_map(self):
        img_to_file = {}
        for safe_file in self.files:
            with safe_open(safe_file, framework="pt", device="cpu") as f:
                labels = f.get_slice('labels')
                labels_shape = labels.get_shape()
                num_imgs = labels_shape[0]
                cur_len = len(img_to_file)
                for i in range(num_imgs):
                    img_to_file[cur_len+i] = {
                        'safe_file': safe_file,
                        'idx_in_file': i
                    }
        return img_to_file

    def __len__(self):
        return len(self.img_to_file_map.keys())

    def __getitem__(self, idx):
        img_info = self.img_to_file_map[idx]
        safe_file, img_idx = img_info['safe_file'], img_info['idx_in_file']
        with safe_open(safe_file, framework="pt", device="cpu") as f:
            ts = f.get_slice('t')
            ys = f.get_slice('y')
            cs = f.get_slice('c')
            xs = f.get_slice('x')
            features = f.get_slice('features')
            labels = f.get_slice('labels')
            t = ts[img_idx:img_idx+1]
            y = ys[img_idx:img_idx+1]
            c = cs[img_idx:img_idx+1]
            x = xs[img_idx:img_idx+1]
            feature = features[img_idx:img_idx+1]
            label = labels[img_idx:img_idx+1]

        # remove the first batch dimension (=1) kept by get_slice()
        t = t.squeeze(0)
        y = y.squeeze(0)
        c = c.squeeze(0)
        x = x.squeeze(0)
        feature = feature.squeeze(0)
        label = label.squeeze(0)
        return t, y, c, x, feature, label
    


if __name__ == "__main__":
    from torch.utils.data import DataLoader
    data_dir = r'/data/checkpoints/LanguageBind/offline_feature/offline_dit_s_feature_256/imagenet_val_256'
    dataset = FeatureDataset(data_dir)
    num_sample_to_vis = 4
    loader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True
    )
    for t, y, c, feature, label in loader:
        print(
            't.shape', t.shape, 'y.shape', y.shape, 'c.shape', c.shape, 
            'feature.shape', feature.shape, 'label.shape', label.shape
            )
        print('label', label)
        break
    