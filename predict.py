import os
import os.path as osp

import numpy as np
import torch
import models
from dataprocess import load_xlsx


class NormalEncoder:
    def __init__(self):
        pass

    def __call__(self, x):
        return torch.cat(
            [
            x,
            -x,
            torch.pow(x, 2),
            torch.pow(x, 3),
            torch.sin(x),
            torch.cos(x),
            torch.sin(2. * x),
            torch.cos(3. * x),
            torch.sin(2. * x),
            torch.cos(3. * x),
            ]
            *
            16,
            dim=-1
        )


class PolynomialEncoder:
    def __init__(self, n):
        self.n = n

    def __call__(self, x):
        return torch.cat(
            ([x] + [torch.pow(x, i) for i in range(1, self.n)]) * 16, dim=-1
        )


class SignalFunction:
    def __init__(self, model_file_path='pths'):
        self.discontinuity_points = [5740, 12933, 12933 + 7077]
        in_channels = 10 * 16
        self.net0 = models.NormalAnn(in_channels=in_channels, h_channels=512 * 4)
        self.net0.load_state_dict(torch.load(osp.join(model_file_path, 'net-patch-0.bin')))
        self.net0.eval()
        in_channels = 8 * 16
        self.net1 = models.DeepAnn1(in_channels=in_channels, h_channels=400)
        self.net1.load_state_dict(torch.load(osp.join(model_file_path, 'net-patch-1.bin')))
        self.net1.eval()
        in_channels = 8 * 16
        self.net2 = models.DeepAnn(in_channels=in_channels, h_channels=512 * 2)
        self.net2.load_state_dict(torch.load(osp.join(model_file_path, 'net-patch-2.bin')))
        self.net2.eval()
        self.encoder1 = NormalEncoder()
        self.encoder2 = PolynomialEncoder(8)
        self.encoder = [self.encoder1, self.encoder2, self.encoder2]
        self.model = [self.net0, self.net1, self.net2]

    def __call__(self, x):
        results = [
            self._call_single_input(x[i]).item() for i in range(len(x))
        ]
        return results

    def auto_diff(self, x):
        out = []
        dif = []
        for i in range(len(x)):
            y, dx = self._call_single_input(x[i], diff=True)
            out.append(y.item())
            dif.append(dx.item())
        return out, dif

    def _call_single_input(self, x, diff=False):
        choice = self._map_to_encoder(x)
        x = torch.tensor([x], dtype=torch.float32)
        x.requires_grad = diff
        # normx = (x - 1.0) / (20010.0 - 1.0) * 25.0
        normx = x
        normx = self.encoder[choice](normx)
        normx = normx.unsqueeze(dim=0)
        y = self.model[choice](normx)
        # y = y / 25.0
        y = y.sum()
        if diff:
            y.backward()
            return y, x.grad.sum()
        else:
            return y

    def _map_to_encoder(self, x):
        if x < ((5740.0 - 1.0) / 20009.0 * 25.0):
            return 0
        elif x < ((12933.0 - 1.0) / 20009.0 * 25.0):
            return 1
        else:
            return 2


if __name__ == '__main__':
    signal_function = SignalFunction()
    x, y = load_xlsx('data/signal.xlsx')
    x = (x - 1.0) / (20010.0 - 1.0) * 25.0
    y = y * 25.0
    x = x[5740:]
    y = y[5740:]
    # out = signal_function([x[23], x[6478], x[13678]])
    # print(out, [y[23], y[6478], y[13678]])
    # out, dif = signal_function.auto_diff([x[23], x[6478], x[13678]])
    # print(out, [y[23], y[6478], y[13678]], dif)
    out, dif = signal_function.auto_diff(x)
    out = np.array(out, dtype=np.float32)
    dif = np.array(dif, dtype=np.float32)
    from matplotlib import pyplot as plt
    plt.plot(x, y)
    plt.plot(x, out)
    plt.plot(x, dif)
    plt.show()
