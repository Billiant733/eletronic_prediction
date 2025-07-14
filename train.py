import torch
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim

from pre_processing.sample_generate import sample_generate, TimeSeriesDataset
from model.lstm import LSTMModel

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def train_model(model, train_loader, val_loader, criterion, optimizer, epochs, patience=5):
    """训练模型并返回最佳模型"""
    model.to(device)
    best_val_loss = float('inf')
    counter = 0
    train_losses, val_losses = [], []

    for epoch in range(epochs):
        # 训练阶段
        model.train()
        train_loss = 0.0
        for inputs, targets in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}"):
            optimizer.zero_grad()
            # print("输入数据：")
            # print(inputs)
            # print(inputs.shape)
            if torch.isnan(inputs).any():
                print("Input contains NaN!")
            if torch.isinf(inputs).any():
                print("Input contains Inf!")

            inputs, targets = inputs.to(device), targets.to(device)

            # 前向传播
            outputs = model(inputs)
            # print("--------------------------------")
            # print(outputs.shape, targets.shape)
            assert outputs.shape == targets.shape, f"形状不匹配: {outputs.shape} vs {targets.shape}"
            loss = criterion(outputs, targets)

            # 反向传播和优化
            # optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * inputs.size(0)

        # 验证阶段
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item() * inputs.size(0)

        # 计算平均损失
        train_loss /= len(train_loader.dataset)
        val_loss /= len(val_loader.dataset)
        train_losses.append(train_loss)
        val_losses.append(val_loss)

        print(f"Epoch {epoch + 1}/{epochs}: Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")

        # 早停检查
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model = model.state_dict()
            counter = 0
        else:
            counter += 1
            if counter >= patience:
                print(f"Early stopping at epoch {epoch + 1}")
                break

    # 加载最佳模型
    model.load_state_dict(best_model)
    return model, train_losses, val_losses


if __name__ == '__main__':
    pass

