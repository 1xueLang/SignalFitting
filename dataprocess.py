import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


def load_xlsx(filename):
    df = pd.read_excel(filename, header=None, index_col=None)
    data_dict = {col: df[col].values for col in df.columns}
    inputs, labels = data_dict[0], data_dict[1]
    return np.array(inputs, dtype=np.float32), np.array(labels, dtype=np.float32)


def normalization(l, scaler=1.):
    _min = min(l)
    _max = max(l)
    return (l - _min) / (_max - _min) * scaler


class SerialData(Dataset):
    def __init__(self, root, norm=(True, False), scaler=(25., 1.), transform=None, target_transform=None, fixed=True):
        super(SerialData, self).__init__()
        self.transform = transform
        self.target_transform = target_transform
        self.inputs, self.labels = load_xlsx(root)
        if norm[0]:
            self.inputs = normalization(self.inputs, scaler[0])
        if norm[1]:
            self.labels = normalization(self.labels, scaler[1])
        if fixed and self.transform is not None:
            self.inputs = np.array(
                [self.transform(x) for x in self.inputs],
                dtype=np.float32
            )
        if fixed and self.target_transform is not None:
            self.labels = np.array(
                [self.target_transform(y) for y in self.labels],
                dtype=np.float32
            )
        self.fixed = fixed

    def __len__(self):
        return len(self.inputs) // 3

    def __getitem__(self, item):
        x = self.inputs[item]
        y = self.labels[item]
        if not self.fixed:
            if self.transform is not None:
                x = self.transform(x)
            if self.target_transform is not None:
                y = self.target_transform(y)
        return x, y


class DataTransform:
    def __init__(self, config={'m': 'normal'}):
        super(DataTransform, self).__init__()
        if config['m'] == 'normal':
            self.funclist = [
                lambda x: x,
                lambda x: -x,
                lambda x: x * x,
                lambda x: np.power(x, 3),
                np.sin,
                np.cos,
                lambda x: np.sin(2 * x),
                lambda x: np.cos(3 * x),
                lambda x: np.sin(2 * x),
                lambda x: np.cos(3 * x),
                # np.exp2,
                # np.exp
            ] * 16
        else:
            self.funclist = [lambda x: np.power(x, i + 1) for i in range(config['n'])]

    def __call__(self, x):
        return np.array(
            [f(x) for f in self.funclist], dtype=np.float32
        )

class TargetTransform:
    def __init__(self, scaler=25.):
        self.scaler=scaler

    def __call__(self, y):
        return y * self.scaler


def get_serial_data(root, batch=-1, config={'m': 'normal'}, shuffle=False):
    ds = SerialData(
        root,
        transform=DataTransform(config),
        target_transform=TargetTransform(),
    )
    batch = len(ds) if batch == -1 else batch
    dl = DataLoader(
        dataset=ds,
        batch_size=batch,
        shuffle=shuffle,
    )
    return dl


if __name__ == '__main__':
    dl = get_serial_data('data/signal.xlsx')

    for x, y in dl:
        print(x)
        print(y)
        