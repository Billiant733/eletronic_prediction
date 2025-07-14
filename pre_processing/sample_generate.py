import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import MinMaxScaler, StandardScaler

import sys
sys.path.append("/home/qa/electronic_prediction/pre_processing")
from preprocessing import processing


def sample_generate(file_path, pred_len, known_len=90, stride=1, is_test=False):
    df = processing(file_path)
    df, scaler = normalize(df, method='minmax', feature_range=(0, 1))
    X_samples, Y_samples = [], []
    for i in range(0, df.shape[0]-pred_len-known_len, stride):
        X_samples.append(df.iloc[i: i+known_len, 1:])   # 不包含时间
        Y_samples.append(df.iloc[i + known_len: i + known_len + pred_len, 1])

        # X_samples.append(normalize(df.iloc[i: i+known_len, 1:]))   # 不包含时间
        # if not is_test:    # 测试集的预测值不需要标准化
        #     new_df = normalize(df.iloc[i+known_len: i+known_len+pred_len, 1:2])
        #     Y_samples.append(new_df.iloc[:, 0])    # 只取第二列（总有功功率），待预测
        # else:
        #     Y_samples.append(df.iloc[i+known_len:i+known_len+pred_len, 1])

    return np.array(X_samples), np.array(Y_samples), scaler


def normalize(df, method='minmax', feature_range=(0, 1)):
    normalized_df = df.copy()
    # print(df)
    for col in df.columns:
        # 检查列是否为数值类型
        if not np.issubdtype(df[col].dtype, np.number):
            print(f"警告: 列 '{col}' 不是数值类型，跳过归一化")
            continue

        # 选择归一化方法
        if method == 'minmax':
            scaler = MinMaxScaler(feature_range=feature_range)
        elif method == 'zscore':
            scaler = StandardScaler()
        else:
            raise ValueError("不支持的归一化方法。请选择 'minmax' 或 'zscore'")

        # 对列进行归一化
        col_data = df[col].values.reshape(-1, 1)   # 将一维数组重塑为二维数组。-1表示自动计算该维度的大小；1表示每行只有一个特征
        normalized_col = scaler.fit_transform(col_data)

        # 保存归一化结果
        normalized_df[col] = normalized_col.flatten()   # 展平为一维

        # 保存归一化器

    return normalized_df, scaler


class TimeSeriesDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


if __name__ == '__main__':
    X_samples, Y_samples, scaler = sample_generate("../data/train_pre.csv", pred_len=90)
    print(X_samples.shape)
    # X_samples = torch.FloatTensor(X_samples).view(-1)    # view(-1)展平为一维向量
    # Y_samples = torch.FloatTensor(Y_samples).view(-1)    # view(-1)展平为一维向量
    # print(X_samples)
