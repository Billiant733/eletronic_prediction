import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
# %matplotlib inline

from pre_processing.sample_generate import sample_generate, TimeSeriesDataset
from model.lstm import LSTMModel
from model.transformer import TransformerModel
from model.dlinear_residual_transformer import WeatherAwareDLinearTransformer
from train import train_model
from evaluate import evaluate_model


def model_train(model, train_dataset, epochs, saved_model_path):
    train_size = int(0.8 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_subset, val_subset = torch.utils.data.random_split(
        train_dataset, [train_size, val_size]
    )

    batch_size = 32
    train_short_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
    val_short_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=True)


    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0005)
    model, _, _ = train_model(
        model, train_short_loader, val_short_loader,
        criterion, optimizer, epochs=epochs
    )
    torch.save(model, saved_model_path)
    print(f"Save model to {saved_model_path}")


def eval_and_show(test_dataset, scaler, which_model, saved_model_path):
    model = torch.load(saved_model_path)
    results_short = {
        'LSTM': {'mse': [], 'mae': []},
        'Transformer': {'mse': [], 'mae': []},
        'D_R_Transformer': {'mse': [], 'mae': []}
    }

    # 评估
    test_short_loader = DataLoader(test_dataset, batch_size=16, shuffle=True)
    criterion = nn.MSELoss()
    _, mse, mae, all_outputs, all_targets = evaluate_model(
        model, test_short_loader, criterion, scaler
    )
    results_short[which_model]['mse'].append(mse)
    results_short[which_model]['mae'].append(mae)
    print(f"{which_model} Test MSE: {mse:.4f}, MAE: {mae:.4f}")
    # print(f"测试集的输入数据尺寸：{X_test.shape}；标准答案的数据尺寸：{Y_test.shape}")
    # print(all_outputs.shape)
    show(all_targets, all_outputs)


def show(ground_truth, actual_predictions):
    pred_len = ground_truth.shape[1]
    plt.title('The day change of global active power')
    plt.ylabel('global_active_power')
    plt.grid(True)
    plt.autoscale(axis='x', tight=True)
    ground_truth = ground_truth.tolist()
    actual_predictions = actual_predictions.tolist()
    show_ground, show_pred = [], []
    # print(ground_truth[0:2])
    for i in range(1, len(ground_truth), 90):
        if pred_len == 90:
            show_ground.append(ground_truth[i - 1] + ground_truth[i])
        else:
            show_ground.append(ground_truth[i-1][90:] + ground_truth[i])
        show_pred.append(actual_predictions[i])
    # print(show_ground)
    for i in range(0, len(show_pred)):
        plt.plot(show_ground[i], color='red')
        x = np.arange(90, 90+pred_len, 1)
        plt.plot(x, show_pred[i], color='deepskyblue')
        plt.show()


if __name__ == '__main__':
    # 加载数据
    # 1. 未来90天预测数据
    print("预测未来90天：数据加载")
    X_train, Y_train, _ = sample_generate("./data/train_pre_minmax.csv", pred_len=90, is_test=False)
    train_short_dataset = TimeSeriesDataset(X_train, Y_train)
    X_test, Y_test, short_scaler = sample_generate("./data/test_pre_minmax.csv", pred_len=90, is_test=True)
    test_short_dataset = TimeSeriesDataset(X_test, Y_test)
    print(f"训练集的输入数据形状：{X_train.shape}；标准数据形状：{Y_train.shape}")
    print(f"测试集的输入数据形状：{X_test.shape}；标准数据形状：{Y_test.shape}")

    # 2. 未来180天预测数据
    print("预测未来180天：数据加载")
    X_train, Y_train, _ = sample_generate("./data/train_pre_minmax.csv", pred_len=180, is_test=False)
    train_long_dataset = TimeSeriesDataset(X_train, Y_train)
    X_test, Y_test, long_scaler = sample_generate("./data/test_pre_minmax.csv", pred_len=180, is_test=True)
    test_long_dataset = TimeSeriesDataset(X_test, Y_test)
    print(f"训练集的输入数据形状：{X_train.shape}；标准数据形状：{Y_train.shape}")
    print(f"测试集的输入数据形状：{X_test.shape}；标准数据形状：{Y_test.shape}")

    input_dim = 12    # 特征数

    # # LSTM模型（90天）
    # pred_len = 90
    # lstm_params = {
    #     'input_dim': input_dim,
    #     'hidden_dim': 128,
    #     'num_layers': 2,
    #     'output_length': pred_len
    # }
    # lstm_short_model = LSTMModel(**lstm_params)
    # model_train(lstm_short_model, train_short_dataset, epochs=150, saved_model_path="./saved_models/lstm_short_model.pt")
    # eval_and_show(test_short_dataset, scaler=short_scaler, which_model='LSTM', saved_model_path="./saved_models/lstm_short_model.pt")

    # # LSTM模型（180天）
    # pred_len = 180
    # lstm_params = {
    #     'input_dim': input_dim,
    #     'hidden_dim': 128,
    #     'num_layers': 2,
    #     'output_length': 180
    # }
    # lstm_long_model = LSTMModel(**lstm_params)
    # model_train(lstm_long_model, train_long_dataset, epochs=150, saved_model_path="./saved_models/lstm_long_model.pt")
    # eval_and_show(test_long_dataset, scaler=short_scaler, which_model='LSTM', saved_model_path="./saved_models/lstm_long_model.pt")

    # # Transformer模型（90天）
    # pred_len = 90
    # transformer_params = {
    #     'input_dim': input_dim,
    #     'embed_dim': 128,
    #     'nhead': 4,
    #     'num_layers': 2,
    #     'output_length': pred_len
    # }
    # transformer_short_model = TransformerModel(**transformer_params)
    # model_train(transformer_short_model, train_short_dataset, epochs=150, saved_model_path="./saved_models/transformer_short_model.pt")
    # eval_and_show(test_short_dataset, short_scaler, "Transformer", "./saved_models/transformer_short_model.pt")

    # # Transformer模型（180天）
    # pred_len = 180
    # transformer_params = {
    #     'input_dim': input_dim,
    #     'embed_dim': 128,
    #     'nhead': 4,
    #     'num_layers': 2,
    #     'output_length': pred_len
    # }
    # transformer_long_model = TransformerModel(**transformer_params)
    # model_train(transformer_long_model, train_long_dataset, epochs=150,
    #             saved_model_path="./saved_models/transformer_long_model.pt")
    # eval_and_show(test_long_dataset, short_scaler, "Transformer", "./saved_models/transformer_long_model.pt")

    # # DLinearResidualTransformer模型（90天）
    # pred_len = 90
    # dr_transformer_params = {
    #     'seq_len': 90,
    #     'pred_len': pred_len,
    # }
    # dr_transformer_short_model = WeatherAwareDLinearTransformer(**dr_transformer_params)
    # model_train(dr_transformer_short_model, train_short_dataset, epochs=150, saved_model_path="./saved_models/dr_transformer_short_model.pt")
    # eval_and_show(test_short_dataset, scaler=short_scaler, which_model='D_R_Transformer',saved_model_path="./saved_models/dr_transformer_short_model.pt")


    # DLinearResidualTransformer模型（180天）
    pred_len = 180
    dr_transformer_params = {
        'seq_len': 90,
        'pred_len': pred_len,
    }
    dr_transformer_long_model = WeatherAwareDLinearTransformer(**dr_transformer_params)
    model_train(dr_transformer_long_model, train_long_dataset, epochs=150,
                saved_model_path="./saved_models/dr_transformer_long_model.pt")
    eval_and_show(test_long_dataset, scaler=long_scaler, which_model='D_R_Transformer',
                  saved_model_path="./saved_models/dr_transformer_long_model.pt")