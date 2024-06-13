import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


class Power:
    def __init__(self, p):
        self.p = p

    def __call__(self, x):
        return np.power(x, self.p)


def load_xlsx(filename):
    df = pd.read_excel(filename, header=None, index_col=None)
    data_dict = {col: df[col].values for col in df.columns}
    inputs, labels = data_dict[0], data_dict[1]
    return np.array(inputs, dtype=np.float32), np.array(labels, dtype=np.float32)


def normalization(l, scaler=1.):
    _min = min(l)
    _max = max(l)
    # print("+", _min, _max, scaler)
    return (l - _min) / (_max - _min) * scaler


class SerialData(Dataset):
    def __init__(self, root, patch=0, norm=(True, False), scaler=(25., 1.), transform=None, target_transform=None, fixed=True, expand=False):
        super(SerialData, self).__init__()
        self.transform = transform
        self.target_transform = target_transform
        self.inputs, self.labels = load_xlsx(root)
        self.patch = patch
        self.base, self.sub_len = self._base_and_len()
        print(self.base, self.sub_len)
        if norm[0]:
            self.inputs = normalization(self.inputs, scaler[0])
        if norm[1]:
            self.labels = normalization(self.labels, scaler[1])
        self.inputs = self.inputs[self.base: self.base + self.sub_len]
        self.labels = self.labels[self.base: self.base + self.sub_len]
        if expand and patch == 1:
            self._get_expand()
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
        return len(self.inputs)

    def __getitem__(self, item):
        # item = item + self.base
        x = self.inputs[item]
        y = self.labels[item]
        if not self.fixed:
            if self.transform is not None:
                x = self.transform(x)
            if self.target_transform is not None:
                y = self.target_transform(y)
        return x, y

    def _base_and_len(self):
        if self.patch == -1:
            return 0, len(self.inputs)
        base = [0, 5740, 12933]
        total_len = len(self.inputs)
        ends = [base[1], base[2], total_len]
        sub_len = [ends[i] - base[i] for i in range(3)]
        base.append(base[1])
        sub_len.append(total_len - base[1])
        return base[self.patch], sub_len[self.patch]

    def _get_expand(self):
        dx = self.inputs[1] - self.inputs[0]
        dy = self.labels[1] - self.labels[0]
        ax = np.array([(self.inputs[0] - dx * i) for i in range(400, 0, -1)], dtype=np.float32)
        ay = np.array([(self.labels[0] - dy * i) for i in range(400, 0, -1)], dtype=np.float32)
        self.inputs = np.append(ax, self.inputs)
        self.labels = np.append(ay, self.labels)

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
        elif config['m'] == 't':
            self.funclist = ([lambda x: x, lambda x: -x] + [Power(i) for i in range(2, config['n'])]) * 16
            # self.funclist = [
            #     lambda x: x,
            #     lambda x: -x,
            #     lambda x: np.power(x, 2),
            #     lambda x: np.power(x, 3),
            #     lambda x: np.power(x, 4),
            #     lambda x: np.power(x, 5),
            #     lambda x: np.power(x, 6),
            #     lambda x: np.power(x, 7),
            # ] * 16
        elif config['m'] == 'f':
            self.funclist = (
                [lambda x: x] +
                [lambda x: np.sin(np.pi * i * x / 9.) for i in range(0, config['n'])] +
                [lambda x: -x] +
                [lambda x: -np.cos(np.pi * i * x / 9.) for i in range(config['n'])] +
                [lambda x: x] +
                [lambda x: -np.sin(np.pi * i * x / 9.) for i in range(0, config['n'])] +
                [lambda x: -x] +
                [lambda x: -np.cos(np.pi * i * x / 9.) for i in range(config['n'])]
            )
    def __call__(self, x):
        return np.array(
            [f(x) for f in self.funclist], dtype=np.float32
        )

class TargetTransform:
    def __init__(self, scaler=25.):
        self.scaler=scaler

    def __call__(self, y):
        return y * self.scaler


def get_serial_data(root, batch=-1, config={'m': 'normal'}, shuffle=False, patch=0, expand=False):
    ds = SerialData(
        root,
        patch=patch,
        transform=DataTransform(config),
        target_transform=TargetTransform(),
        expand=expand
    )
    batch = len(ds) if batch == -1 else batch
    if isinstance(batch, float):
        batch = int(len(ds) * batch)
    dl = DataLoader(
        dataset=ds,
        batch_size=batch,
        shuffle=shuffle,
    )
    return dl


if __name__ == '__main__':
    dl = get_serial_data('data/signal.xlsx', config={'m': 't', 'n': 8}, shuffle=False, patch = 1)
    for x, y in dl:
        print(x)
        # print(y)
        