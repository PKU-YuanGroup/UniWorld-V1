
import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import random
import os
from time import time
from tqdm import tqdm
import argparse
import matplotlib.pyplot as plt
import seaborn as sns

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from datasets.feature_dataset import FeatureDataset

def set_seed(seed=42):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_size, num_classes):
        super(MLP, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, num_classes)
        )

    @torch.compile
    def forward(self, x):
        return self.model(x)

def evaluate_model(data_key, feature_layer_idx, model, val_loader, criterion, device, train_step=0):
    model.eval()
    val_loss = 0
    correct = 0
    confusion_matrix = torch.zeros(1000, 1000, dtype=torch.int64)  # 1000x1000 矩阵

    with torch.no_grad():
        for step, (t, y, c, x, feature, label) in enumerate(tqdm(val_loader)):
            data_batch_dict = dict(t=t, y=y, c=c, x=x, feature=feature[:, feature_layer_idx])
            x_batch = data_batch_dict[data_key].float().to(device, non_blocking=True)
            y_batch = label.to(device, non_blocking=True)
            output = model(x_batch)
            
            val_loss += criterion(output, y_batch).item()
            pred = output.argmax(dim=1)  # 取最大值索引作为预测类别
            correct += (pred == y_batch).sum().item()

            # 更新 confusion matrix
            for i in range(y_batch.size(0)):
                true_label = y_batch[i].item()
                pred_label = pred[i].item()
                confusion_matrix[true_label, pred_label] += 1
    import ipdb;ipdb.set_trace()
    std = sum([i.float().std().item() for i in confusion_matrix]) / 1000
    avg_loss = val_loss / len(val_loader)
    accuracy = correct / len(val_loader.dataset)
    print(f"Validation Loss: {avg_loss:.8f}, Accuracy: {accuracy:.4f}")

    # 绘制热力图
    plt.figure(figsize=(10, 8))
    sns.heatmap(confusion_matrix.cpu().numpy(), cmap="Blues", norm=None)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix")
    plt.savefig(f"{data_key}-{feature_layer_idx}-{std}-{train_step}.jpg")

def train_model(data_key, feature_layer_idx, model, train_loader, val_loader, criterion, optimizer, epochs=10, log_interval=100):
    eval_interval = log_interval * 10
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        start_time = time()
        for step, (t, y, c, x, feature, label) in enumerate(train_loader):
            data_batch_dict = dict(t=t, y=y, c=c, x=x, feature=feature[:, feature_layer_idx])
            x_batch = data_batch_dict[data_key].float().to(device, non_blocking=True)
            y_batch = label.to(device, non_blocking=True)
            optimizer.zero_grad()
            output = model(x_batch)
            loss = criterion(output, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
            if step % log_interval == 0:
                end_time = time()
                speed = log_interval / (end_time - start_time) 
                print(f"Epoch [{epoch+1}/{epochs}], Step [{step}/{len(train_loader)}], Loss: {total_loss/log_interval:.8f}, Step/sec: {speed:.4f}")
                total_loss = 0
                start_time = time()
            if step != 0 and step % eval_interval == 0:
                evaluate_model(data_key, feature_layer_idx, model, val_loader, criterion, device, step)
        evaluate_model(data_key, feature_layer_idx, model, val_loader, criterion, device, step)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default='/path/to/your/data')
    parser.add_argument("--data_key", type=str, default='feature')
    parser.add_argument("--feature_layer_idx", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    
    set_seed(args.seed)
    dataset = FeatureDataset(args.data_path)
    train_size = int(0.99 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    print('len(train_dataset)', len(train_dataset), 'len(val_dataset)', len(val_dataset))
    train_loader = DataLoader(train_dataset, batch_size=1024, num_workers=32, pin_memory=True, prefetch_factor=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=1024, num_workers=32, pin_memory=True, prefetch_factor=16, shuffle=False)
    
    num_features, hidden_size, num_classes = 384, 2048, 1000
    model = MLP(num_features, hidden_size, num_classes)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.0)
    
    epochs, log_interval = 1, 100
    train_model(args.data_key, int(args.feature_layer_idx), model, train_loader, val_loader, criterion, optimizer, epochs, log_interval)
