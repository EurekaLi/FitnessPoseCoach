"""PYSKL ST-GCN++ 的单人、三分类纯 PyTorch 适配，不是新的算法。

依据 PYSKL 的 STGCN、unit_gcn、mstcn 和 COCO spatial 图；保留 10 阶段、
可学习初始邻接矩阵、图内残差及六分支时间卷积。修改了输入排列、分类数与数据流程，
去掉 MMCV 注册/配置依赖。出处与许可见 third_party/PYSKL_NOTICE.md。
窗口内卷积使用双侧填充，但窗口完全截止于当前时刻；不输出窗口内部逐帧预测。
"""

from __future__ import annotations

import torch
from torch import nn


def coco_adjacency() -> torch.Tensor:
    """COCO17 的自连接、向内、向外三个子图，按列归一化，方向与 PYSKL 一致。"""
    inward = [(15, 13), (13, 11), (16, 14), (14, 12), (11, 5), (12, 6),
              (9, 7), (7, 5), (10, 8), (8, 6), (5, 0), (6, 0),
              (1, 0), (3, 1), (2, 0), (4, 2)]
    matrices = [torch.eye(17)]
    for links in (inward, [(j, i) for i, j in inward]):
        matrix = torch.zeros(17, 17)
        for i, j in links:
            matrix[j, i] = 1
        matrices.append(matrix / matrix.sum(dim=0).clamp_min(1))
    return torch.stack(matrices)


class TemporalConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel: int = 1,
                 stride: int = 1, dilation: int = 1, norm: bool = True):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, (kernel, 1),
                              stride=(stride, 1), padding=(dilation * (kernel - 1) // 2, 0),
                              dilation=(dilation, 1))
        self.bn = nn.BatchNorm2d(out_channels) if norm else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.bn(self.conv(x))


class SpatialGraphConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        # adaptive='init'：训练直接更新整张初始图，而不只是固定骨骼边的乘法权重。
        self.A = nn.Parameter(coco_adjacency())
        self.conv = nn.Conv2d(in_channels, out_channels * 3, 1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.down = (nn.Identity() if in_channels == out_channels else
                     nn.Sequential(nn.Conv2d(in_channels, out_channels, 1), nn.BatchNorm2d(out_channels)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        n, _, t, v = x.shape
        features = self.conv(x).reshape(n, 3, -1, t, v)
        aggregated = torch.einsum("nkctv,kvw->nctw", features, self.A).contiguous()
        return torch.relu(self.bn(aggregated) + self.down(x))


class MultiScaleTemporalConv(nn.Module):
    def __init__(self, channels: int, stride: int, dropout: float):
        super().__init__()
        branch_width = channels // 6
        widths = [channels - 5 * branch_width] + [branch_width] * 5
        branches = []
        # 四种时间间隔让同一层观察快慢不同的运动；剩余通道分给首分支以保留总宽度。
        for width, dilation in zip(widths[:4], (1, 2, 3, 4)):
            branches.append(nn.Sequential(nn.Conv2d(channels, width, 1), nn.BatchNorm2d(width), nn.ReLU(),
                                          TemporalConv(width, width, 3, stride, dilation, norm=False)))
        branches.append(nn.Sequential(nn.Conv2d(channels, widths[4], 1), nn.BatchNorm2d(widths[4]), nn.ReLU(),
                                      nn.MaxPool2d((3, 1), stride=(stride, 1), padding=(1, 0))))
        branches.append(nn.Conv2d(channels, widths[5], 1, stride=(stride, 1)))
        self.branches = nn.ModuleList(branches)
        self.transform = nn.Sequential(nn.BatchNorm2d(channels), nn.ReLU(), nn.Conv2d(channels, channels, 1))
        self.bn = nn.BatchNorm2d(channels)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.bn(self.transform(torch.cat([branch(x) for branch in self.branches], dim=1))))


class STGCNBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int,
                 residual: bool, dropout: float):
        super().__init__()
        self.gcn = SpatialGraphConv(in_channels, out_channels)
        self.tcn = MultiScaleTemporalConv(out_channels, stride, dropout)
        self.residual = None if not residual else (nn.Identity() if in_channels == out_channels and stride == 1
                                                  else TemporalConv(in_channels, out_channels, stride=stride))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = 0 if self.residual is None else self.residual(x)
        return torch.relu(self.tcn(self.gcn(x)) + residual)


class STGCNPP(nn.Module):
    """输入 (B,3,48,17)，输出 (B,3) 未归一化分类分数；三个通道为 x/y/置信度。

    只对完整的过去窗口做一个判断。时间卷积不是可逐帧复用隐藏状态的流式卷积。
    """

    def __init__(self, num_classes: int = 3, dropout: float = 0.1):
        super().__init__()
        self.data_bn = nn.BatchNorm1d(17 * 3)
        blocks, current = [], 3
        for stage in range(1, 11):
            channels = 64 if stage < 5 else 128 if stage < 8 else 256
            blocks.append(STGCNBlock(current, channels, 2 if stage in (5, 8) else 1,
                                     residual=stage != 1, dropout=0 if stage == 1 else dropout))
            current = channels
        self.blocks = nn.Sequential(*blocks)
        self.fc = nn.Linear(256, num_classes)
        nn.init.normal_(self.fc.weight, std=0.01)
        nn.init.zeros_(self.fc.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        n, c, t, v = x.shape
        # PYSKL 的输入 BN 排列是 (joint, channel)，不是直接合并 (channel, joint)。
        x = x.permute(0, 3, 1, 2).contiguous().reshape(n, v * c, t)
        x = self.data_bn(x).reshape(n, v, c, t).permute(0, 2, 3, 1).contiguous()
        return self.fc(self.blocks(x).mean(dim=(2, 3)))
