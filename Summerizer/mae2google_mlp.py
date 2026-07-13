import os
import h5py
import math
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm


# ===============================
# AdaptMLP 定义
# ===============================
class AdaptMLP(nn.Module):
    def __init__(
        self,
        hidden_dim,
        prenorm=False,
        midnorm=False,
        norm_fn=nn.LayerNorm,
        act_fn=nn.GELU,
        scale=1.0,
        zinit=False,
        dim=None,
        output_dim=None,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.prenorm = prenorm
        self.midnorm = midnorm
        self.norm_fn = norm_fn
        self.act_fn = act_fn
        self.scale = nn.Parameter(torch.ones(1).float()) if abs(scale) < 1e-6 else scale
        self.zinit = zinit
        if dim is not None:
            self.setup(dim, output_dim)

    def extra_repr(self):
        return f"scale={self.scale}, zinit={self.zinit}"

    def setup(self, dim, output_dim=None):
        layers = []

        if self.prenorm:
            layers.append(self.norm_fn(dim))

        layers.append(nn.Linear(dim, self.hidden_dim))
        if self.zinit:
            nn.init.kaiming_uniform_(layers[-1].weight, a=math.sqrt(5))
            nn.init.zeros_(layers[-1].bias)

        if self.midnorm:
            layers.append(self.norm_fn(self.hidden_dim))

        layers.append(self.act_fn())

        layers.append(
            nn.Linear(self.hidden_dim, dim if output_dim is None else output_dim)
        )
        if self.zinit:
            nn.init.zeros_(layers[-1].weight)
            nn.init.zeros_(layers[-1].bias)

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.scale * self.layers(x)


# ===============================
# StableFeatureMapper 替换版：使用 AdaptMLP
# ===============================
class StableFeatureMapper(nn.Module):
    """
    用 AdaptMLP 替代原线性层，实现 VideoMAE(1024) → GoogLeNet(1024) 映射
    """
    def __init__(self, input_dim=1024, output_dim=1024, hidden_dim=2048, dropout=0.1):
        super().__init__()

        # 使用 AdaptMLP 作为主映射层
        self.mapper = AdaptMLP(
            hidden_dim=hidden_dim,
            prenorm=True,
            midnorm=True,
            norm_fn=nn.LayerNorm,
            act_fn=nn.GELU,
            scale=1.0,
            zinit=False,
            dim=input_dim,
            output_dim=output_dim,
        )

        # 残差连接，线性投影
        self.shortcut = nn.Linear(input_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(output_dim)

    def forward(self, x):
        out = self.mapper(x)
        out = self.dropout(out)
        out = out + self.shortcut(x)
        return self.norm(out)


# ===============================
# 读取 VideoMAE 和 GoogLeNet 特征
# ===============================
def load_features(input_mae_h5, input_google_h5):
    pairs = []
    with h5py.File(input_mae_h5, 'r') as f_m, h5py.File(input_google_h5, 'r') as f_g:
        for video_id in tqdm(f_m.keys(), desc="Loading features from H5"):
            if video_id not in f_g:
                print(f"[WARN] {video_id} not in GoogLeNet h5, skip.")
                continue

            m_group = f_m[video_id]
            g_group = f_g[video_id]

            if 'mae_frame_feature' not in m_group or 'features' not in g_group:
                continue

            mae_feat = torch.tensor(m_group['mae_frame_feature'][:], dtype=torch.float32)
            google_feat = torch.tensor(g_group['features'][:], dtype=torch.float32)
            pairs.append((mae_feat, google_feat))
    return pairs


# ===============================
# 训练函数
# ===============================
def train_mapper(model, data_pairs, epochs=100, lr=1e-4, device='cuda', patience=8, min_delta=1e-4):
    model = model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.MSELoss()

    best_loss = float('inf')
    counter = 0
    best_state = None

    for epoch in range(epochs):
        total_loss = 0.0
        model.train()

        for mae_feat, google_feat in tqdm(data_pairs, desc=f"Epoch {epoch+1}/{epochs}"):
            mae_feat, google_feat = mae_feat.to(device), google_feat.to(device)
            optimizer.zero_grad()

            mapped = model(mae_feat)

            mse_loss = criterion(mapped, google_feat)
            reg_loss = 1e-3 * torch.mean((mapped - mae_feat[:, :mapped.shape[1]]) ** 2)
            loss = mse_loss + reg_loss

            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(data_pairs)
        print(f"Epoch {epoch+1} | Avg Loss: {avg_loss:.6f}")

        # Early Stopping
        if avg_loss + min_delta < best_loss:
            best_loss = avg_loss
            best_state = model.state_dict()
            counter = 0
            print(f"✅ Loss improved to {best_loss:.6f}")
        else:
            counter += 1
            print(f"⚠️ No improvement for {counter} epoch(s)")
            if counter >= patience:
                print(f"\n⏹️ Early stopping triggered at epoch {epoch+1}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    return model


# ===============================
# 写入输出 h5
# ===============================
def save_with_mapped_features(input_mae_h5, output_h5, model, device='cuda'):
    with h5py.File(input_mae_h5, 'r') as fin, h5py.File(output_h5, 'w') as fout:
        for video_id in tqdm(fin.keys(), desc="Saving mapped features"):
            fin.copy(video_id, fout)
            mae_feat = torch.tensor(fin[video_id]['mae_frame_feature'][:], dtype=torch.float32).to(device)
            with torch.no_grad():
                mapped_feat = model(mae_feat).cpu().numpy()
            fout[video_id].create_dataset('mapped_feature', data=mapped_feat)


# ===============================
# 主流程
# ===============================
if __name__ == "__main__":
    input_mae_h5 = "/home/ma-user/work/code/pred_h5/eccv16_dataset_summe_google_pool5_mae_frame_feature_1024_attnpool.h5"
    input_google_h5 = "/home/ma-user/work/code/CSTA/data/datasets/eccv16_dataset_summe_google_pool5.h5"
    output_h5 = "/home/ma-user/work/code/pred_h5/eccv16_dataset_summe_google_pool5_mae2google_adaptmlp_1024_attnpool.h5"

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    # 1. 加载数据
    data_pairs = load_features(input_mae_h5, input_google_h5)

    # 2. 初始化模型（AdaptMLP版）
    mapper = StableFeatureMapper(input_dim=1024, output_dim=1024, hidden_dim=2048)

    # 3. 训练映射器
    mapper = train_mapper(mapper, data_pairs, epochs=500, lr=1e-4, device=device)

    # 4. 保存映射结果
    save_with_mapped_features(input_mae_h5, output_h5, mapper, device=device)
    print(f"\n✅ Done! AdaptMLP-based mapping saved to: {output_h5}")
