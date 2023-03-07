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
        x = x / th.max(x) * self.scale
        max_t, batch_size, channel = x.size()
        mem = th.zeros((batch_size, channel))
        record = []
        for t in range(max_t):
            mem = (mem + x[t]) * self.decay
            out = 1 * (mem > 1)
            mem = mem - out
            record.append(out)
        return th.stack(record)
