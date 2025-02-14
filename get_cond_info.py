# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DiT: https://github.com/facebookresearch/DiT
# LightningDiT: https://github.com/hustvl/LightningDiT
# --------------------------------------------------------

import os
import math
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from models.did import DiD_models


import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

def plot_hist_with_normal_curve(data, bins=30, i=0):
    plt.figure(figsize=(8, 5))

    # 画直方图
    count, bins_edge, _ = plt.hist(data, bins=bins, density=True, alpha=0.6, color='g', edgecolor='black')

    # 计算均值和标准差
    mu, sigma = np.mean(data), np.std(data)

    # 叠加正态分布曲线
    x = np.linspace(min(data), max(data), 100)
    plt.plot(x, stats.norm.pdf(x, mu, sigma), 'r', linewidth=2)

    # 标题和标签
    plt.title(f"Histogram with Normal Curve (μ={mu:.2f}, σ={sigma:.2f})")
    plt.xlabel("Value")
    plt.ylabel("Density")

    plt.savefig(f'{i}.jpg')


if __name__ == '__main__':
    import yaml
    from datasets.img_latent_dataset import ImgLatentDataset

    config_path = 'configs/did_s_100kx1024_qf1x1_img1p0.yaml'
    with open(config_path, "r") as f:
        train_config = yaml.safe_load(f)
    # Create model:
    if 'downsample_ratio' in train_config['vae']:
        downsample_ratio = train_config['vae']['downsample_ratio']
    else:
        downsample_ratio = 8
    assert train_config['data']['image_size'] % downsample_ratio == 0, "Image size must be divisible by 8 (for the VAE encoder)."
    latent_size = train_config['data']['image_size'] // downsample_ratio
    in_channels = train_config['model']['in_chans'] if 'in_chans' in train_config['model'] else 4
    use_diffusion = train_config['scheduler']['diffusion'] if 'diffusion' in train_config['scheduler'] else False
    model = DiD_models[train_config['model']['model_type']](
        input_size=latent_size,
        num_classes=train_config['data']['num_classes'],
        use_qknorm=train_config['model']['use_qknorm'],
        use_swiglu=train_config['model']['use_swiglu'] if 'use_swiglu' in train_config['model'] else False,
        use_rope=train_config['model']['use_rope'] if 'use_rope' in train_config['model'] else False,
        use_rmsnorm=train_config['model']['use_rmsnorm'] if 'use_rmsnorm' in train_config['model'] else False,
        in_channels=in_channels, 
        use_checkpoint=train_config['model']['use_checkpoint'] if 'use_checkpoint' in train_config['model'] else False,
        qformer=train_config['model']['qformer'] if 'qformer' in train_config['model'] else None,
        learn_sigma=train_config['diffusion']['learn_sigma'] if use_diffusion and 'learn_sigma' in train_config['diffusion'] else False,
    ).cuda()
    checkpoint = torch.load(train_config['ckpt_path'], map_location=lambda storage, loc: storage)
    checkpoint['model'] = {k.replace('module.', ''): v for k, v in checkpoint['model'].items()}
    model.load_state_dict(checkpoint['model'])
    model.eval()
    # print(model)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")


    latent_norm = train_config['data']['latent_norm'] if 'latent_norm' in train_config['data'] else False
    dataset = ImgLatentDataset(
        data_dir=train_config['data']['data_path'],
        latent_norm=latent_norm,
        latent_multiplier=train_config['data']['latent_multiplier'] if 'latent_multiplier' in train_config['data'] else 0.18215,
    )
    bsz = 64
    loader = DataLoader(
        dataset,
        batch_size=bsz,
        shuffle=True,
        num_workers=16,
        pin_memory=True,
        drop_last=False
    )

    # logits = {i: [] for i in range(1000)}
    # for i, (clean_x, y) in enumerate(tqdm(loader)):
    #     clean_x = clean_x.cuda()
    #     y = y.cuda()
    #     # logit = model.y_embedder.proj(clean_x).flatten(2).transpose(1, 2).detach().cpu()    # (B, N, D)
    #     logit = model.q_former(model.y_embedder(clean_x, None, False, B=bsz, num_tokens=256)).detach().cpu()
    #     for idx, l in enumerate(y):
    #         logits[int(l)].append(logit[idx])
    #     # if i > 1000:
    #     #     break
    '''
    -0.008968254551291466 0.25275373458862305 5.179049491882324 -6.220407485961914
    -0.7921016812324524 1.4527924060821533 3.2326178550720215 -16.556354522705078
    '''
    # logits = {k: torch.stack(v) for k, v in logits.items()} # class: B D
    # cnt = 0
    # for k, v in logits.items():
    #     for i in range(10): # vis first 10 channel
    #         plot_hist_with_normal_curve(v[:, i].numpy().flatten(), bins=100, i=f'{k}_c{i}')
    #     cnt += 1
    #     if cnt == 2:  # vis 2 classes
    #         break
    # mean = [torch.mean(logits[i], dim=0) for i in range(1000)]
    # std = [torch.std(logits[i], dim=0) for i in range(1000)]
    # print(mean[:8][:8], std[:8][:8])
    # torch.save(dict(mean=mean, std=std), 'meta_info.pt')

    # nn.init.normal_(model.y_embedder.embedding_table.weight, mean=-0.008968254551291466, std=0.25275373458862305)
    logits = []
    for i, (clean_x, y) in enumerate(loader):
        clean_x = clean_x.cuda()
        y = y.cuda()
        # logit = model.y_embedder(clean_x, None, False, B=bsz, num_tokens=256).detach().cpu()
        logit = model.q_former(model.y_embedder(clean_x, None, False, B=bsz, num_tokens=256)).detach().cpu()
        print('clean_x', logit.mean().item(), logit.std().item(), logit.max().item(), logit.min().item())
        # logit_y = model.y_embedder(None, y, False, B=bsz, num_tokens=256).detach().cpu()
        # logit_y = model.q_former(model.y_embedder(None, y, False, B=bsz, num_tokens=256)).detach().cpu()
        # print('label_y', logit_y.mean().item(), logit_y.std().item(), logit_y.max().item(), logit_y.min().item())
        if i > 1:
            break


    print('zero_qformer')

    train_config['model']['qformer']['zero_qformer'] = True
    model = DiD_models[train_config['model']['model_type']](
        input_size=latent_size,
        num_classes=train_config['data']['num_classes'],
        use_qknorm=train_config['model']['use_qknorm'],
        use_swiglu=train_config['model']['use_swiglu'] if 'use_swiglu' in train_config['model'] else False,
        use_rope=train_config['model']['use_rope'] if 'use_rope' in train_config['model'] else False,
        use_rmsnorm=train_config['model']['use_rmsnorm'] if 'use_rmsnorm' in train_config['model'] else False,
        in_channels=in_channels, 
        use_checkpoint=train_config['model']['use_checkpoint'] if 'use_checkpoint' in train_config['model'] else False,
        qformer=train_config['model']['qformer'] if 'qformer' in train_config['model'] else None,
        learn_sigma=train_config['diffusion']['learn_sigma'] if use_diffusion and 'learn_sigma' in train_config['diffusion'] else False,
    ).cuda()
    checkpoint = torch.load(train_config['ckpt_path'], map_location=lambda storage, loc: storage)
    checkpoint['model'] = {k.replace('module.', ''): v for k, v in checkpoint['model'].items()}
    model.load_state_dict(checkpoint['model'])
    model.eval()

    meta_info = torch.load('/data/FlowWorld/meta_info.pt')
    mean, std = meta_info['mean'], meta_info['std']
    generated_vectors = torch.stack(mean)
    print('generated_vectors', generated_vectors.shape)
    with torch.no_grad():
        model.y_embedder.embedding_table.weight.data[:-1].copy_(generated_vectors)
    logits = []
    for i, (clean_x, y) in enumerate(loader):
        clean_x = clean_x.cuda()
        y = y.cuda()
        # logit = model.y_embedder(clean_x, None, False, B=bsz, num_tokens=256).detach().cpu()
        # logit = model.q_former(model.y_embedder(clean_x, None, False, B=bsz, num_tokens=256)).detach().cpu()
        # print('clean_x', logit.mean().item(), logit.std().item(), logit.max().item(), logit.min().item())
        # logit_y = model.y_embedder(None, y, False, B=bsz, num_tokens=256).detach().cpu()
        logit_y = model.q_former(model.y_embedder(None, y, False, B=bsz, num_tokens=256)).detach().cpu()
        print('label_y', logit_y.mean().item(), logit_y.std().item(), logit_y.max().item(), logit_y.min().item())
        if i > 1:
            break