import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class WeatherAwareDLinearTransformer(nn.Module):
    def __init__(self, seq_len=90, pred_len=90,
                 elec_feat_dim=7, weather_feat_dim=5, d_model=128, nhead=4):
        super().__init__()

        self.pred_len = pred_len

        # ===== 1. 电力特征预处理 =====
        self.elec_proj = nn.Linear(elec_feat_dim, d_model // 2)

        # ===== 2. 天气特征专项处理 =====
        # 降水特征处理（RR需要除以10）
        self.rain_encoder = nn.Sequential(
            nn.Linear(1, 8),
            nn.GELU(),
            nn.Linear(8, 16))

        # 降水天数特征处理（NBJRR1/5/10）
        self.rain_days_encoder = nn.Sequential(
            nn.Linear(3, 16),
            nn.GELU())

        # 雾天特征处理
        self.fog_encoder = nn.Sequential(
            nn.Linear(1, 8),
            nn.GELU())

        # 天气特征融合
        self.weather_fusion = nn.Linear(16 + 16 + 8, d_model // 2)

        # ===== 3. 时序处理主干 =====
        # DLinear分支
        self.trend_proj = nn.Linear(seq_len, d_model)
        self.seasonal_proj = nn.Linear(seq_len, d_model)

        # Transformer分支
        self.pos_enc = PositionalEncoding(d_model)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward=4 * d_model),
            num_layers=3)

        # ===== 4. 特征重要性加权 =====
        self.attention_weights = nn.Parameter(torch.ones(2))  # 电力vs天气权重

        # ===== 5. 输出层 =====
        self.output = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, pred_len))

    def forward(self, x):
        """
        x_elec: [batch, seq_len, 7] (电力特征)
        x_weather: [batch, seq_len, 5] (天气特征顺序: RR, NBJRR1, NBJRR5, NBJRR10, NBJBROU)
        """
        # ===== 天气特征专项处理 =====
        # 降水高度处理 (RR/10)
        x_elec = x[..., :7]
        x_weather = x[..., 7:]
        rr = x_weather[..., 0:1] / 10.0  # [batch, seq_len, 1]
        rain_emb = self.rain_encoder(rr)

        # 降水天数处理
        rain_days = x_weather[..., 1:4]  # [batch, seq_len, 3]
        days_emb = self.rain_days_encoder(rain_days)

        # 雾天处理
        fog = x_weather[..., 4:5]  # [batch, seq_len, 1]
        fog_emb = self.fog_encoder(fog)

        # 天气特征融合
        weather_feat = torch.cat([rain_emb, days_emb, fog_emb], dim=-1)
        weather_feat = self.weather_fusion(weather_feat)  # [batch, seq_len, d_model//2]

        # ===== 电力特征处理 =====
        elec_feat = self.elec_proj(x_elec)  # [batch, seq_len, d_model//2]

        # ===== 特征重要性加权 =====
        feat_weights = F.softmax(self.attention_weights, dim=0)
        combined = torch.cat([
            elec_feat * feat_weights[0],
            weather_feat * feat_weights[1]
        ], dim=-1)  # [batch, seq_len, d_model]

        # ===== DLinear分支 =====
        trend = combined.mean(dim=-1)  # [batch, seq_len]
        seasonal = combined - trend.unsqueeze(-1)

        trend_pred = self.trend_proj(trend)  # [batch, pred_len]
        seasonal_pred = self.seasonal_proj(seasonal.mean(dim=-1))

        # ===== Transformer分支 =====
        x_trans = self.pos_enc(combined)
        trans_out = self.transformer(x_trans)[:, -1, :]  # 取最后pred_len步
        # print()
        # print(trend_pred.shape, seasonal_pred.shape, trans_out.shape)
        # print(trend_pred.unsqueeze(-1).shape)

        # ===== 输出融合 =====
        output = self.output(trans_out + trend_pred + seasonal_pred)
        return output.squeeze(-1)


# 位置编码（同前）
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(1), :].unsqueeze(0)