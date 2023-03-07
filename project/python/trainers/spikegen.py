import torch as th

"""
Author: Arthur Wang
Date: Mar 6
"""


class SpikeGenerator:

    def __init__(self, scale, decay):
        self.scale = scale
        self.decay = decay

    def transform(self, _, x):
        max_t, batch_size, channel = x.size()
        x = (x / th.amax(x, (0, 2)).expand((max_t, channel, batch_size)).swapaxes(1, 2)) * self.scale
        mem = th.zeros((batch_size, channel))
        record = []
        for t in range(max_t):
            mem = (mem + x[t]) * self.decay
            out = 1 * (mem > 1)
            mem = mem - out
            record.append(out)
        return th.stack(record)
