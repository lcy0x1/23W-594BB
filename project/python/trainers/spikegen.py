import torch as th

"""
Author: Arthur Wang
Date: Mar 6
"""


class SpikeGenerator:

    def __init__(self, step, scale, sp, sn, decay):
        self.step = step
        self.scale = scale
        self.sp = sp
        self.sn = sn
        self.decay = decay

    def transform(self, _, x):
        x = x.permute(2, 0, 1)
        max_t, batch_size, channel = x.size()
        x = th.clamp((x + self.sn) / (self.sp + self.sn), 0, 1) * self.scale
        mem = th.zeros((batch_size, channel))
        record = []
        for t in range(max_t):
            mem = (mem + x[t]) * self.decay
            out = 1 * (mem > 1)
            mem = mem - out
            record.append(out)
        ans = th.concat((th.stack(record), th.zeros((self.step - max_t, batch_size, channel))), dim=0)
        return ans
