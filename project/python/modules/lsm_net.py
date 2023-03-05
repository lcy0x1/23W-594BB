from typing import List

import torch
from torch import nn

from generator import SignalSource, NeuronDataInput, NeuronDataImpl, NeuronConnection
from linear_leak import LinearLeakLIF
from lsm_hyperparam import STDPLearner, LSMInitializer

"""
Author: Arthur Wang
Date: Mar 4
"""


class LSMNeuron(LinearLeakLIF):

    def __init__(self, beta=1.0, threshold=1.0):
        super().__init__(beta, threshold, None, False, False)


class LSMPool(nn.Module):

    def __init__(self, in_size, hidden_size, threshold, init: LSMInitializer, stdp: STDPLearner):
        super().__init__()
        # Initialize layers
        self.in_size = in_size
        self.hidden_size = hidden_size
        self.total_size = in_size + hidden_size
        self.fc1 = nn.Linear(self.total_size, self.hidden_size, bias=False)
        self.lif = LSMNeuron(threshold=threshold)
        init.init_lsm_conn(self.fc1)
        self.stdp = stdp.stdp

    def forward(self, x):
        """
        Please call with torch.no_grad() to prevent gradient calculation
        :param x: Tensor with size (time_step, batch_size, input_size)
        :return:
        """

        batch_size = x.size(1)
        max_t = self.optm_param.num_steps
        assert x.size() == (max_t, batch_size, self.in_size)

        # Initialize hidden states at t=0
        mem = self.lif.init_leaky()

        # Record the final layer
        spk_rec = []
        mem_rec = []

        # output spike for hidden layers
        spk_hidden = torch.zeros((x.size(1), self.hidden_size))  # (batch_size, neuron_size)
        # time since previous spike
        spk_time = torch.zeros((x.size(1), self.total_size)) + max_t
        # merged spike
        spk = torch.concat((x[0], spk_hidden), dim=1)
        # initialize spike time
        if self.training:
            spk_time = (spk_time + 1) * (1 - spk)

        for step in range(max_t):
            # pass through weights
            cur = self.fc1(spk)
            # generate spike
            spk_hidden, mem = self.lif(cur, mem)
            # record spikes
            spk_rec.append(spk_hidden)
            mem_rec.append(mem)
            # get next spike now, for training purpose
            if step + 1 == max_t:
                spk_in = torch.zeros((x.size(1), self.in_size))
            else:
                spk_in = x[step + 1]
            spk = torch.concat((spk_in, spk_hidden), dim=1)

            # STDP implementation
            if self.training:
                spk_time = (spk_time + 1) * (1 - spk)
                for pre in range(self.total_size):
                    if round(spk_time[pre].item()) == 0:
                        time_slice = spk_time[self.in_size:self.total_size]
                        self.fc1.weight.T[pre] = self.stdp(self.fc1.weight.T[pre], -time_slice)
                for post in range(self.hidden_size):
                    if round(spk_time[self.in_size + post].item()) == 0:
                        self.fc1.weight[post] = self.stdp(self.fc1.weight[post], spk_time)

        return torch.stack(spk_rec, dim=0).detach(), torch.stack(mem_rec, dim=0)

    def generate(self) -> List[SignalSource]:
        ans = []
        for i in range(self.in_size):
            ans.append(NeuronDataInput())
        for i in range(self.hidden_size):
            leak = round(self.lif.beta[i].item())
            thres = round(self.lif.threshold[i].item())
            ans.append(NeuronDataImpl(leak, thres))
        for i in range(self.hidden_size):
            for j in range(self.in_size + self.hidden_size):
                ws = round(self.fc1.weight[j][self.in_size + i].item())
                # TODO regulate weights
                if ws != 0:
                    ans[self.in_size + i].add_conn(NeuronConnection(ans[j], ws))
        return ans
