import torch
import torch.nn as nn
import numpy as np


class TransformerModel(nn.Module):
    def __init__(self, input_dim, embed_dim, nhead=3, num_layers=6, output_length=90):
        super(TransformerModel, self).__init__()
        self.d_model = embed_dim

        # 输入嵌入层
        self.input_fc = nn.Linear(input_dim, embed_dim)

        # 位置编码
        self.positional_encoding = PositionalEncoding(embed_dim)

        # Transformer编码器
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=nhead,
            dim_feedforward=4 * embed_dim,
            dropout=0.2,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # 输出层
        self.fc_out = nn.Linear(embed_dim, output_length)

    def forward(self, x):
        # 输入嵌入
        x = self.input_fc(x)

        # 添加位置编码
        x = self.positional_encoding(x)

        # Transformer编码
        x = self.transformer_encoder(x)
        # print(x.shape)

        # 取最后一个时间步
        x = x[:, -1, :]
        # print(x.shape)

        # 输出层
        output = self.fc_out(x)
        return output


# 位置编码
class PositionalEncoding(nn.Module):
    def __init__(self, embed_dim, max_len=5000):
        super(PositionalEncoding, self).__init__()

        pe = torch.zeros(max_len, embed_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2).float() * (-np.log(10000.0) / embed_dim))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]
