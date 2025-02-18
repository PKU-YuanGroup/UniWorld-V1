from sklearn.linear_model import SGDClassifier
from torch.utils.data import DataLoader

import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, Subset
import random
import os
from time import time
from tqdm import tqdm
import argparse
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from datasets.feature_dataset import FeatureDataset

def set_seed(seed=42):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default='/path/to/your/data')
    parser.add_argument("--data_key", type=str, default='feature')
    parser.add_argument("--feature_layer_idx", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n_jobs", type=int, default=16)
    args = parser.parse_args()
    
    set_seed(args.seed)
    dataset = FeatureDataset(args.data_path)
    train_size = int(0.99 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    print('len(train_dataset)', len(train_dataset), 'len(val_dataset)', len(val_dataset))
    train_loader = DataLoader(train_dataset, batch_size=1024, num_workers=32, pin_memory=True, prefetch_factor=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=1024, num_workers=32, pin_memory=True, prefetch_factor=16, shuffle=False)

    clf = SGDClassifier(loss="log_loss", max_iter=1, tol=1e-3, n_jobs=args.n_jobs) 

    # train
    for idx, (t, y, c, x, feature, label) in enumerate(tqdm(train_loader)):
        data_batch_dict = dict(t=t, y=y, c=c, x=x, feature=feature[:, args.feature_layer_idx])
        x_batch = data_batch_dict[args.data_key].float()
        clf.partial_fit(x_batch.numpy(), label.numpy(), classes=list(range(1000)))  # 1000 类别
        if idx == 100:
             break

    # eval
    correct = 0
    total = 0
    for idx, (t, y, c, x, feature, label) in enumerate(tqdm(val_loader)):
        data_batch_dict = dict(t=t, y=y, c=c, x=x, feature=feature[:, args.feature_layer_idx])
        x_batch = data_batch_dict[args.data_key].float()
        y_pred = clf.predict(x_batch.numpy())
        correct += (y_pred == label.numpy()).sum()
        total += len(label)
        if idx == 100:
             break

    print(f"Validation Accuracy: {correct / total:.4f}")
