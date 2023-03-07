import torch as th

"""
Author: Arthur Wang
Date: Mar 6
"""


class SpikeGenerator:

    def __init__(self, scale, sigma, decay):
        self.scale = scale
        self.sigma = sigma
        self.decay = decay

    def transform(self, _, x):
        x = x.swapaxis(0, 1)
        max_t, batch_size, channel = x.size()
        x = th.clamp((x / self.sigma + 1) / 2, 0, 1) * self.scale
        mem = th.zeros((batch_size, channel))
        record = []
        for t in range(max_t):
            mem = (mem + x[t]) * self.decay
            out = 1 * (mem > 1)
            mem = mem - out
            record.append(out)
        return th.stack(record)
