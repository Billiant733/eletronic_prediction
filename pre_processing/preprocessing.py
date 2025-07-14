import torch
import pandas as pd
import random
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler


def processing(file_path, new_file_path=""):
    df = pd.read_csv(file_path)
    # df = pd.read_csv(file_path, names=['time', 'global_active_power', 'global_reactive_power', 'sub_metering_1', 'sub_metering_2', 'sub_metering_3', 'RR', 'NBJRR1', 'NBJRR5', 'NBJRR10', 'NBJBROU'])

    df['date'] = df.iloc[:, 0].str.split(' ').str[0]

    # 缺失值处理
    # df.dropna(axis=0, how='any', inplace=True)    # 删掉缺失值；缺失值填充在下方使用

    # df['sub_metering_remainder'] = (
    #         df.iloc[:, 1] * 1000 * 24 -
    #         (df.iloc[:, 5] +
    #          df.iloc[:, 6] +
    #          df.iloc[:, 7])
    # )
    # 按日期分组
    grouped = df.groupby('date')
    days_data = []
    for date, group in grouped:
        # print(f"\n日期: {date}")
        # print(f"记录数: {len(group)}")
        # print(group.head(2))
        day_data = [date]
        day_data.extend(min2day(group))
        days_data.append(day_data)
    new_df = pd.DataFrame(days_data, columns=['date', 'global_active_power', 'global_reactive_power', 'voltage', 'global_intensity', 'sub_metering_1', 'sub_metering_2', 'sub_metering_3', 'RR', 'NBJRR1', 'NBJRR5', 'NBJRR10', 'NBJBROU'])
    # new_df = normalize(new_df)

    if torch.isnan(torch.tensor(np.array(new_df.iloc[:, 1:]), dtype=torch.float32)).any():
        print("Input contains NaN!")
    # new_df.to_csv(new_file_path, index=False)
    # print(f"write down to file {new_file_path}")
    return new_df


def min2day(day_df):
    new_data = []
    for i in [1, 2, 5, 6, 7,]:     # 求和的列：global_active_power、global_reactive_power、sub_metering_1、sub_metering_2
        df_ = pd.to_numeric(day_df.iloc[:, i], errors='coerce')
        df_.ffill(axis=0, inplace=True)  # 用前一个非缺失值填充
        df_.bfill(axis=0, inplace=True)  # 用后一个非缺失值填充
        df_.fillna(0, inplace=True)  # 此外若某整列缺失，则用0填充
        new_data.append(df_.sum())
    for j in [3, 4]:
        df_ = pd.to_numeric(day_df.iloc[:, j], errors='coerce')
        df_.ffill(axis=0, inplace=True)  # 用前一个非缺失值填充
        df_.bfill(axis=0, inplace=True)  # 用后一个非缺失值填充
        df_.fillna(0, inplace=True)  # 此外若某整列缺失，则用0填充
        new_data.append(df_.mean())
    for k in range(8, 13):
        df_ = pd.to_numeric(day_df.iloc[:, k], errors='coerce')
        df_.ffill(axis=0, inplace=True)  # 用前一个非缺失值填充
        df_.bfill(axis=0, inplace=True)  # 用后一个非缺失值填充
        df_.fillna(0, inplace=True)  # 此外若某整列缺失，则用0填充
        new_data.append(df_.iloc[random.randint(0, len(day_df)-1)])
    return new_data


def normalize(df, method='minmax', feature_range=(0, 1)):
    normalized_df = df.copy()

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

    return normalized_df


if __name__ == '__main__':
    # pre_processing("../data/test.csv", "../data/test_pre_minmax.csv")
    print("Done.")
