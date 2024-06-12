import torch
import dataprocess
import models
from matplotlib import pyplot as plt


def print_curve(dl, net, p):
    x, y = next(iter(dl))
    net.eval()
    out = net(x)
    out = out.squeeze()
    net.train()
    x = x.numpy()[:, 0]
    y = y.numpy()
    out = out.detach().numpy()
    plt.clf()
    plt.plot(x, y)
    plt.plot(x, out)
    plt.savefig(f'fimage-{p}.pdf')


def print_loss(e, loss, net):
    print(e, ': ', loss)


if __name__ == '__main__':
    # patch = 0
    # bound = 1e-4
    # shuffle = False
    # batch_size = 6
    # lrs = [1e-3] * 4 + [3e-4] * 6 + [1e-4] * 20
    # root = 'data/signal.xlsx'
    # in_channels = 10 * 16
    # dl = dataprocess.get_serial_data(root, batch_size, config={'m': 'normal'}, shuffle=shuffle, patch=patch)
    # gdl = dataprocess.get_serial_data(root, batch=-1, config={'m': 'normal'}, patch=patch)
    #
    # net = models.NormalAnn(in_channels=in_channels, h_channels=512 * 4)

    # #################################################################################################################################
    patch = 3
    bound = -1.
    shuffle = True
    batch_size = 6
    lrs = [1e-5] * 100

    root = 'data/signal.xlsx'
    in_channels = 8 * 16
    dl = dataprocess.get_serial_data(root, batch_size, config={'m': 't', 'n': in_channels // 16}, shuffle=shuffle, patch=patch)
    gdl = dataprocess.get_serial_data(root, batch=-1, config={'m': 't', 'n': in_channels // 16}, patch=patch)
    # in_channels = 10 * 16
    # dl = dataprocess.get_serial_data(root, batch_size, config={'m': 'normal'}, shuffle=shuffle, patch=patch)
    # gdl = dataprocess.get_serial_data(root, batch=-1, config={'m': 'normal'}, patch=patch)
    net = models.NormalAnn(in_channels=in_channels, h_channels=512 * 4)
    net.load_state_dict(torch.load('pths/net-patch-3.bin'))
    # ##################################################################################################################################
    epochs = 128 * 8
    net.train()
    criterion = torch.nn.MSELoss()
    # optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    optimizer = torch.optim.SGD(net.parameters(), lr=1., momentum=0.9)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: lrs[-1] if epoch >= 450 else lrs[epoch // 50])
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimezer, T_max=64)

    best = 99999.0
    for e in range(epochs):
        for i, (x, y) in enumerate(dl):
            out = net(x).squeeze()
            loss = criterion(out, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss = loss.item()
            if 1e-3 < loss < best:
                best = min(best, loss)
                print_curve(gdl, net, patch)
                torch.save(net.state_dict(), f'pths/net-patch-{patch}.bin')
            if i % 50 == 0:
                print(scheduler.get_last_lr(), e, loss, best)
        scheduler.step()
