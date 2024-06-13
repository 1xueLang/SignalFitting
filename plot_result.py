import torch
import models
import dataprocess
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd


def load_xlsx(filename):
    df = pd.read_excel(filename, header=None, index_col=None)
    data_dict = {col: df[col].values for col in df.columns}
    inputs, labels = data_dict[0], data_dict[1]
    return np.array(inputs, dtype=np.float32), np.array(labels, dtype=np.float32)


if __name__ == '__main__':
    in_channels = 10 * 16
    net0 = models.NormalAnn(in_channels=in_channels, h_channels=512 * 4)
    net0.load_state_dict(torch.load('pths/net-patch-0.bin'))
    net0.eval()
    in_channels = 8 * 16
    net1 = models.DeepAnn1(in_channels=in_channels, h_channels=400)
    net1.load_state_dict(torch.load('pths/net-patch-1.bin'))
    net1.eval()
    in_channels = 8 * 16
    net2 = models.DeepAnn(in_channels=in_channels, h_channels=512 * 2)
    net2.load_state_dict(torch.load('pths/net-patch-2.bin'))
    net2.eval()

    root = 'data/signal.xlsx'
    patch0 = dataprocess.get_serial_data(root, batch=-1, config={'m': 'normal'}, patch=0)
    patch1 = dataprocess.get_serial_data(root, batch=-1, config={'m': 't', 'n': 8}, patch=1)
    patch2 = dataprocess.get_serial_data(root, batch=-1, config={'m': 't', 'n': 8}, patch=2)

    x0, y0 = next(iter(patch0))
    x1, y1 = next(iter(patch1))
    x2, y2 = next(iter(patch2))

    out0 = net0(x0).squeeze()
    out1 = net1(x1).squeeze()
    out2 = net2(x2).squeeze()

    x = np.append(x0[:, 0].detach().numpy(), x1[:, 0].detach().numpy())
    x = np.append(x, x2[:, 0].detach().numpy())

    y = np.append(y0.detach().numpy(), y1.detach().numpy())
    y = np.append(y, y2.detach().numpy())

    out = np.append(out0.detach().numpy(), out1.detach().numpy())
    out = np.append(out, out2.detach().numpy())

    plt.plot(x, y)
    plt.plot(x, out)
    plt.savefig('fimage.pdf')