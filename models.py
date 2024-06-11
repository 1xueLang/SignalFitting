import torch
import torch.nn as nn


class NormalAnn(nn.Module):
    def __init__(self, in_channels, h_channels=1024):
        super(NormalAnn, self).__init__()
        AF = nn.Sigmoid
        # NM = nn.LayerNorm
        NM = nn.BatchNorm1d
        self.nn = nn.Sequential(
            nn.Linear(in_channels, h_channels),
            # nn.BatchNorm1d(h_channels),
            NM(h_channels),
            AF(),
            nn.Linear(h_channels, h_channels // 2),
            # nn.BatchNorm1d(h_channels // 2),
            NM(h_channels // 2),
            AF(),
            nn.Linear(h_channels // 2, h_channels // 4),
            # nn.BatchNorm1d(h_channels // 4),
            NM(h_channels // 4),
            AF(),
            nn.Linear(h_channels // 4, 10)
        )

    def forward(self, inputs):
        h = self.nn(inputs)
        return h.mean(dim=-1)