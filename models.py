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


class DeepAnn(nn.Module):
    def __init__(self, in_channels, h_channels=1024):
        super(DeepAnn, self).__init__()
        AF = nn.Sigmoid
        # NM = nn.LayerNorm
        NM = nn.BatchNorm1d
        self.ih = nn.Sequential(
            nn.Linear(in_channels, h_channels // 2),
            NM(h_channels // 2),
            AF()
        )
        self.nn1 = nn.Sequential(
            nn.Linear(h_channels // 2, h_channels // 2),
            # nn.BatchNorm1d(h_channels),
            NM(h_channels // 2),
            AF(),
            nn.Linear(h_channels // 2, h_channels // 2),
            # nn.BatchNorm1d(h_channels // 2),
            NM(h_channels // 2),
            AF(),
        )
        self.nn2 = nn.Sequential(
            nn.Linear(h_channels // 2, h_channels // 2),
            # nn.BatchNorm1d(h_channels // 2),
            NM(h_channels // 2),
            AF(),
            nn.Linear(h_channels // 2, h_channels // 2),
            # nn.BatchNorm1d(h_channels // 2),
            NM(h_channels // 2),
            AF(),
        )
        self.nn3 = nn.Sequential(
            nn.Linear(h_channels // 2, h_channels // 2),
            # nn.BatchNorm1d(h_channels // 2),
            NM(h_channels // 2),
            AF(),

            nn.Linear(h_channels // 2, h_channels // 2),
            # nn.BatchNorm1d(h_channels // 2),
            NM(h_channels // 2),
            AF(),
        )
        self.nn4 = nn.Sequential(
            nn.Linear(h_channels // 2, h_channels // 2),
            # nn.BatchNorm1d(h_channels // 2),
            NM(h_channels // 2),
            AF(),

            nn.Linear(h_channels // 2, h_channels // 2),
            # nn.BatchNorm1d(h_channels // 2),
            NM(h_channels // 2),
            AF(),
        )
        self.nn5 = nn.Sequential(
            nn.Linear(h_channels // 2, h_channels // 2),
            # nn.BatchNorm1d(h_channels // 4),
            NM(h_channels // 2),
            AF(),
            nn.Linear(h_channels // 2, 16)
        )

    def forward(self, inputs):
        h0 = self.ih(inputs)
        h1 = h0 + self.nn1(h0)
        h2 = h0 + self.nn2(h1)
        h3 = h2 + self.nn3(h2)
        h4 = h2 + self.nn4(h3)
        h5 = self.nn5(h4)
        return h5.mean(dim=-1)


class DeepAnn1(nn.Module):
    def __init__(self, in_channels, h_channels=1024):
        super(DeepAnn1, self).__init__()
        AF = nn.Sigmoid
        # NM = nn.LayerNorm
        NM = nn.BatchNorm1d
        self.ih = nn.Sequential(
            nn.Linear(in_channels, h_channels // 2),
            NM(h_channels // 2),
            AF()
        )
        self.nn1 = nn.Sequential(
            nn.Linear(h_channels // 2, h_channels // 2),
            # nn.BatchNorm1d(h_channels),
            NM(h_channels // 2),
            AF(),
            nn.Linear(h_channels // 2, h_channels // 2),
            # nn.BatchNorm1d(h_channels // 2),
            NM(h_channels // 2),
            AF(),
        )
        self.nn2 = nn.Sequential(
            nn.Linear(h_channels // 2, h_channels // 2),
            # nn.BatchNorm1d(h_channels // 2),
            NM(h_channels // 2),
            AF(),
            nn.Linear(h_channels // 2, h_channels // 2),
            # nn.BatchNorm1d(h_channels // 2),
            NM(h_channels // 2),
            AF(),
        )
        self.nn3 = nn.Sequential(
            nn.Linear(h_channels // 2, h_channels // 2),
            # nn.BatchNorm1d(h_channels // 2),
            NM(h_channels // 2),
            AF(),

            nn.Linear(h_channels // 2, h_channels // 2),
            # nn.BatchNorm1d(h_channels // 2),
            NM(h_channels // 2),
            AF(),
        )
        self.nn4 = nn.Sequential(
            nn.Linear(h_channels // 2, h_channels // 2),
            # nn.BatchNorm1d(h_channels // 2),
            NM(h_channels // 2),
            AF(),

            nn.Linear(h_channels // 2, h_channels // 2),
            # nn.BatchNorm1d(h_channels // 2),
            NM(h_channels // 2),
            AF(),
        )
        self.nn5 = nn.Sequential(
            nn.Linear(h_channels // 2, h_channels // 2),
            # nn.BatchNorm1d(h_channels // 2),
            NM(h_channels // 2),
            AF(),

            nn.Linear(h_channels // 2, h_channels // 2),
            # nn.BatchNorm1d(h_channels // 2),
            NM(h_channels // 2),
            AF(),
        )
        # self.nn6 = nn.Sequential(
        #     nn.Linear(h_channels // 2, h_channels // 2),
        #     # nn.BatchNorm1d(h_channels // 2),
        #     NM(h_channels // 2),
        #     AF(),
        #
        #     nn.Linear(h_channels // 2, h_channels // 2),
        #     # nn.BatchNorm1d(h_channels // 2),
        #     NM(h_channels // 2),
        #     AF(),
        # )
        self.nn7 = nn.Sequential(
            nn.Linear(h_channels // 2, h_channels // 2),
            # nn.BatchNorm1d(h_channels // 4),
            NM(h_channels // 2),
            AF(),
            nn.Linear(h_channels // 2, 16)
        )

    def forward(self, inputs):
        h0 = self.ih(inputs)
        h1 = h0 + self.nn1(h0)
        h2 = h1 + self.nn2(h1)
        h3 = h2 + self.nn3(h2)
        h4 = h3 + self.nn4(h3)
        h5 = h4 + self.nn5(h4)
        # h6 = h5 + self.nn6(h5)
        h7 = self.nn7(h5 + h4 + h3 + h2 + h1 + h0)
        return h7.mean(dim=-1)
