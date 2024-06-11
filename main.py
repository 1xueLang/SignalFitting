import torch
import dataprocess
import models
from matplotlib import pyplot as plt


def print_curve(dl, net):
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
    plt.savefig('fimage.pdf')


def print_loss(e, loss, net):
    print(e, ': ', loss)


if __name__ == '__main__':
    root = 'data/signal.xlsx'
    batch_size = 6
    lr = 1e-4
    epochs = 64 * 100000

    # in_channels = 20
    # dl = dataprocess.get_serial_data(root, batch_size, config={'m': 't', 'n': in_channels})
    # gdl = dataprocess.get_serial_data(root, batch=-1, config={'m': 't', 'n': in_channels})
    in_channels = 10 * 16
    dl = dataprocess.get_serial_data(root, batch_size, config={'m': 'normal'}, shuffle=False)
    gdl = dataprocess.get_serial_data(root, batch=-1, config={'m': 'normal'})

    net = models.NormalAnn(in_channels=in_channels, h_channels=512 * 4)
    net.train()
    criterion = torch.nn.MSELoss()
    # optimezer = torch.optim.Adam(net.parameters(), lr=lr)
    optimezer = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimezer, T_max=64)

    best = 99999.0
    for e in range(epochs):
        if e == 1:
            dl = dataprocess.get_serial_data(root, batch=-1, config={'m': 'normal'}, shuffle=False)
        for i, (x, y) in enumerate(dl):
            out = net(x).squeeze()
            loss = criterion(out, y)
            optimezer.zero_grad()
            loss.backward()
            optimezer.step()

            best = min(best, loss.item())
            print_curve(gdl, net)

            if i % 50 == 0:
                print(e, loss.item(), best)
        scheduler.step()