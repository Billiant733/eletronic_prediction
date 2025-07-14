import torch
import torch.nn as nn


class LSTMModel(nn.Module):
    def __init__(self, input_dim=12, hidden_dim=128, num_layers=1, output_length=90):
        super(LSTMModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(hidden_dim, output_length)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def forward(self, x):
        # 初始化隐藏状态
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(self.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(self.device)

        # LSTM前向传播
        out, _ = self.lstm(x, (h0, c0))
        # print("lstm层输出：")
        # print(out)

        # 只取最后一个时间步的输出
        out = out[:, -1, :]
        # print("只取最后一个时间步的输出")
        # print(out)
        # print(out.shape)

        out = self.dropout(out)
        out = self.fc(out)
        # print("经过fc层后的输出")
        # print(out)
        # print(out.shape)

        return out


if __name__ == '__main__':
    model = LSTMModel()
    loss_function = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    print(model)
