"""5-layer Conv1D CNN — 复现 Borkenhagen 2024 / Scarafoni 2019 架构 (PyTorch)"""

import torch
import torch.nn as nn

from config import CONV_FILTERS, CONV_KERNEL, POOL_SIZE, POOL_STRIDE, DENSE_UNITS, DROPOUT, N_AMINO_ACIDS


class BorkenhagenCNN(nn.Module):
    """5 层 Conv1D + MaxPool + BatchNorm → Flatten → Dense → Dropout → Sigmoid"""

    def __init__(self, input_channels=N_AMINO_ACIDS, input_length=581):
        super().__init__()

        filters = CONV_FILTERS  # [32, 64, 128, 256, 512]
        self.conv_blocks = nn.ModuleList()
        self.pools = nn.ModuleList()
        self.batch_norms = nn.ModuleList()

        in_ch = input_channels
        cur_len = input_length

        for f in filters:
            block = nn.Sequential(
                nn.Conv1d(in_ch, f, kernel_size=CONV_KERNEL, padding="same"),
                nn.ReLU(),
            )
            self.conv_blocks.append(block)

            self.pools.append(nn.MaxPool1d(POOL_SIZE, stride=POOL_STRIDE))
            self.batch_norms.append(nn.BatchNorm1d(f))

            in_ch = f
            cur_len = cur_len // POOL_STRIDE

        # 计算 Flatten 后维度
        self.flat_dim = filters[-1] * cur_len

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.flat_dim, DENSE_UNITS),
            nn.ReLU(),
            nn.Dropout(DROPOUT),
            nn.Linear(DENSE_UNITS, 1),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        """
        x: (batch, 21, 581)
        """
        for conv, pool, bn in zip(self.conv_blocks, self.pools, self.batch_norms):
            x = conv(x)   # Conv1D → ReLU
            x = pool(x)   # MaxPool1D
            x = bn(x)     # BatchNorm
        return self.classifier(x)

    def freeze_conv(self):
        """冻结所有卷积 + pool + BN 层（Stage 2 微调用）"""
        for param in self.conv_blocks.parameters():
            param.requires_grad = False
        for param in self.pools.parameters():
            param.requires_grad = False
        for param in self.batch_norms.parameters():
            param.requires_grad = False

    def unfreeze_all(self):
        """解冻所有层"""
        for param in self.parameters():
            param.requires_grad = True


def build_model(device="cuda"):
    """创建模型并移到指定设备"""
    model = BorkenhagenCNN()
    model = model.to(device)
    n_params = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"模型参数量: {n_params:,} (可训练: {n_trainable:,})")
    return model


if __name__ == "__main__":
    model = build_model("cpu")
    x = torch.randn(4, 21, 581)
    y = model(x)
    print(f"Input:  {x.shape}")
    print(f"Output: {y.shape}")  # (4, 1)
    print(model)
