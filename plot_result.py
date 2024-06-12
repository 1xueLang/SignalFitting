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
    net0.train()
    # net1 = models.NormalAnn(in_channels=in_channels, h_channels=512 * 4)
    # net1.load_state_dict(torch.load('pths/net-patch-0.bin'))
    # net2 = models.NormalAnn(in_channels=in_channels, h_channels=512 * 4)
    # net2.load_state_dict(torch.load('pths/net-patch-0.bin'))
    in_channels = 8 * 16
    net3 = models.NormalAnn(in_channels=in_channels, h_channels=512 * 4)
    net3.load_state_dict(torch.load('pths/net-patch-3.bin'))
    net3.train()

    root = 'data/signal.xlsx'
    patch0 = dataprocess.get_serial_data(root, batch=-1, config={'m': 'normal'}, patch=0)
    patch3 = dataprocess.get_serial_data(root, batch=-1, config={'m': 't', 'n': 8}, patch=3)

    x0, y0 = next(iter(patch0))
    x3, y3 = next(iter(patch3))

    out0 = net0(x0).squeeze()
    out3 = net3(x3).squeeze()

    x = np.append(x0[:, 0].detach().numpy(), x3[:, 0].detach().numpy())
    y = np.append(y0.detach().numpy(), y3.detach().numpy())
    out = np.append(out0.detach().numpy(), out3.detach().numpy())

    plt.plot(x, y)
    plt.plot(x, out)
    plt.savefig('fimage.pdf')