import math

import torch


def compress(x):
    n = x.size(0)
    assert math.floor(n / 8) * 8 == n
    size = x.size()
    x = x.reshape((math.floor(n / 8), 8, *size[1:]))
    x = x.swapaxes(0, 1)
    y = 0
    for i in range(8):
        y = y * 2 + x[i]
    return y


def decompress(x):
    n = x.size(0)
    ans = []
    for i in range(8):
        ans.append(x & 1)
        x = x >> 1
    ans.reverse()
    return torch.stack(ans).swapaxes(0, 1).reshape((n * 8, *x.size()[1:]))
