import torch
from torch import nn

from modules.verilog_generator import *
from modules.linear_leak import LinearLeakLIF
from modules.lsm_hyperparam import LSMInitParams, STDPLearner, LSMInitializer

"""
Author: Arthur Wang
Date: Mar 4
"""


class LSMNeuron(LinearLeakLIF):

    def __init__(self, beta=1.0, threshold=1.0, state_quant=torch.relu):
        super().__init__(beta, threshold, None, False, False, state_quant=state_quant)


class LSMPool(nn.Module):

    def __init__(self, optm, param: LSMInitParams, init: LSMInitializer, stdp: STDPLearner):
        super().__init__()
        # Initialize layers
        self.in_size = param.in_size
        self.hidden_size = param.hidden_size
        self.out_size = param.out_size
        self.total_size = self.in_size + self.hidden_size
        self.optm_param = optm

        self.fc1 = nn.Linear(self.total_size, self.hidden_size, bias=False)
        self.lsm = LSMNeuron(beta=1.0, threshold=init.get_lsm_threshold())
        self.fc2 = nn.Linear(self.hidden_size, self.out_size)
        self.readout = LinearLeakLIF(beta=1.0, threshold=init.get_readout_threshold(),
                                     spike_grad=optm.grad, learn_beta=False,
                                     learn_threshold=True)

        init.init_lsm_conn(self.fc1)
        init.init_readout_weight(self.fc2)
        self.stdp = stdp.stdp
        self.lsm_learning = True

    def forward(self, x):
        """
        Please call with torch.no_grad() to prevent gradient calculation
        :param x: Tensor with size (time_step, batch_size, input_size)
        :return:
        """

        self.lsm_learning = False
        batch_size = x.size(1)
        max_t = self.optm_param.num_steps
        assert x.size() == (max_t, batch_size, self.in_size)
        with torch.no_grad():
            x, _ = self._forward_lsm(max_t, batch_size, x)
        x, _ = self._forward_readout(max_t, x)
        return x

    def lsm_train(self, x):
        self.lsm_learning = True
        batch_size = x.size(1)
        max_t = self.optm_param.num_steps
        assert x.size() == (max_t, batch_size, self.in_size)
        with torch.no_grad():
            self._forward_lsm(max_t, batch_size, x)

    def _forward_lsm(self, max_t, batch_size, x):

        # Initialize hidden states at t=0
        mem1 = self.lsm.init_leaky()

        # Record the final layer
        spk_rec = []
        mem_rec = []

        # output spike for hidden layers
        spk_hidden = torch.zeros((batch_size, self.hidden_size))  # (batch_size, neuron_size)
        # time since previous spike
        spk_time = torch.zeros((batch_size, self.total_size)) + max_t
        # merged spike
        spk = torch.concat((x[0], spk_hidden), dim=1)
        # initialize spike time
        if self.lsm_learning:
            spk_time = (spk_time + 1) * (1 - spk)

        for step in range(max_t):
            # pass through weights
            cur = self.fc1(spk)
            # generate spike
            spk_hidden, mem1 = self.lsm(cur, mem1)
            # record spikes
            spk_rec.append(spk_hidden)
            mem_rec.append(mem1)
            # get next spike now, for training purpose
            if step + 1 == max_t:
                spk_in = torch.zeros((batch_size, self.in_size))
            else:
                spk_in = x[step + 1]
            spk = torch.concat((spk_in, spk_hidden), dim=1)

            # STDP implementation
            if self.lsm_learning:
                spk_time = (spk_time + 1) * (1 - spk)
                for batch in range(batch_size):
                    self._perform_stdp(spk_time[batch])

        return torch.stack(spk_rec, dim=0), torch.stack(mem_rec, dim=0)

    def _forward_readout(self, max_t, x):

        # Initialize hidden states at t=0
        mem2 = self.readout.init_leaky()

        # Record the final layer
        spk_rec = []
        mem_rec = []

        for step in range(max_t):
            # pass through weights
            cur = self.fc2(x[step])
            # generate spike
            spk, mem2 = self.lsm(cur, mem2)
            # record spikes
            spk_rec.append(spk)
            mem_rec.append(mem2)

        return torch.stack(spk_rec, dim=0), torch.stack(mem_rec, dim=0)

    def _perform_stdp(self, spk_time):
        for pre in range(self.total_size):
            if round(spk_time[pre].item()) == 0:
                time_slice = spk_time[self.in_size:self.total_size]
                self.fc1.weight.T[pre] = self.stdp(self.fc1.weight.T[pre], -time_slice)
        for post in range(self.hidden_size):
            if round(spk_time[self.in_size + post].item()) == 0:
                self.fc1.weight[post] = self.stdp(self.fc1.weight[post], spk_time)

    def generate(self) -> List[SignalSource]:
        ans = []
        for i in range(self.in_size):
            ans.append(NeuronDataInput())
        for i in range(self.hidden_size):
            leak = round(self.lsm.beta[i].item())
            thres = round(self.lsm.threshold[i].item())
            ans.append(NeuronHidden(leak, thres))
        for i in range(self.out_size):
            leak = round(self.readout.beta[i].item())
            thres = round(self.readout.threshold[i].item())
            ans.append(NeuronReadout(leak, thres))
        for i in range(self.hidden_size):
            for j in range(self.in_size + self.hidden_size):
                ws = round(self.fc1.weight[j][self.in_size + i].item())
                if ws != 0:
                    ans[self.in_size + i].add_conn(NeuronConnection(ans[j], ws))
        for i in range(self.out_size):
            for j in range(self.hidden_size):
                ws = round(self.fc1.weight[j][i].item())
                if ws != 0:
                    ans[self.in_size + self.hidden_size + i].add_conn(NeuronConnection(ans[self.in_size + j], ws))
        return ans
