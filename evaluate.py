import torch
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def evaluate_model(model, test_loader, criterion, y_scaler):
    """评估模型性能"""
    model.eval()
    test_loss = 0.0
    all_targets = []
    all_outputs = []

    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)

            # print(f"outputs形状: {outputs.shape}, targets形状: {targets.shape}")
            assert outputs.shape == targets.shape, f"形状不匹配: {outputs.shape} vs {targets.shape}"
            loss = criterion(outputs, targets)
            test_loss += loss.item() * inputs.size(0)

            # 保存结果用于后续计算
            all_outputs.append(outputs.cpu().numpy())
            all_targets.append(targets.cpu().numpy())

    # 计算平均损失
    test_loss /= len(test_loader.dataset)

    # 合并结果
    all_outputs = np.vstack(all_outputs)
    all_targets = np.vstack(all_targets)

    # 反标准化
    all_outputs = y_scaler.inverse_transform(all_outputs)
    all_targets = y_scaler.inverse_transform(all_targets)

    # 计算指标
    mse = mean_squared_error(all_targets, all_outputs)
    mae = mean_absolute_error(all_targets, all_outputs)

    return test_loss, mse, mae, all_outputs, all_targets



