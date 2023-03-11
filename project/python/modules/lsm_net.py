import torch
from torch import nn

from modules.linear_leak import LinearLeakLIF
from modules.lsm_hyperparam import LSMInitParams, STDPLearner, LSMInitializer
from modules.verilog_generator import *

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
        self.num_steps = optm.num_steps

        # self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.device = torch.device("cpu")

        self.fc1 = nn.Linear(self.total_size, self.hidden_size, bias=False).to(self.device)
        self.lsm = LSMNeuron(beta=torch.ones((self.hidden_size,)).to(self.device),
                             threshold=init.get_lsm_threshold()).to(self.device)
        self.fc2 = nn.Linear(self.hidden_size, self.out_size).to(self.device)
        self.readout = LinearLeakLIF(beta=torch.ones((self.out_size,)), threshold=init.get_readout_threshold(),
                                     spike_grad=optm.grad, learn_beta=False,
                                     learn_threshold=True).to(self.device)

        init.init_lsm_conn(self.fc1)
        init.init_readout_weight(self.fc2)
        self.stdp = stdp
        self.lsm_learning = True

    def forward(self, x):
        """
        Please call with torch.no_grad() to prevent gradient calculation
        :param x: Tensor with size (time_step, batch_size, input_size)
        :return:
        """

        self.lsm_learning = False
        batch_size = x.size(1)
        max_t = self.num_steps
        assert x.size() == (max_t, batch_size, self.in_size)
        with torch.no_grad():
            x, _ = self._forward_lsm(max_t, batch_size, x)
        x, _ = self._forward_readout(max_t, x)
        return x

    def lsm_train(self, x):
        self.lsm_learning = True
        batch_size = x.size(1)
        max_t = self.num_steps
        assert x.size() == (max_t, batch_size, self.in_size)
        with torch.no_grad():
            self._forward_lsm(max_t, batch_size, x)

    def _forward_lsm(self, max_t, batch_size, x):

        # Initialize hidden states at t=0
        mem1 = self.lsm.init_leaky()
        x = x.to(self.device)

        # Record the final layer
        spk_rec = []
        mem_rec = []

        # output spike for hidden layers
        spk_hidden = torch.zeros((batch_size, self.hidden_size)).to(self.device)  # (batch_size, neuron_size)
        # time since previous spike
        spk_time = (torch.zeros((batch_size, self.total_size)) + max_t).to(self.device)
        # merged spike
        spk = torch.concat((x[0], spk_hidden), dim=1).to(self.device)
        # initialize spike time
        if self.lsm_learning:
            spk_time = (spk_time + 1) * (1 - spk)

        for step in range(max_t):
            # pass through weights
            cur = self.fc1(spk)
            # generate spike
            spk_hidden, mem1 = self.lsm(cur, mem1)
            spk_hidden = spk_hidden.to(self.device)
            # record spikes
            spk_rec.append(spk_hidden)
            mem_rec.append(mem1)
            # get next spike now, for training purpose
            if step + 1 == max_t:
                spk_in = torch.zeros((batch_size, self.in_size)).to(self.device)
            else:
                spk_in = x[step + 1]
            spk = torch.concat((spk_in, spk_hidden), dim=1).to(self.device)

            # STDP implementation
            if self.lsm_learning:
                spk_time = (spk_time + 1) * (1 - spk)
                self._perform_stdp(spk_time)

        return torch.stack(spk_rec, dim=0).to(self.device), torch.stack(mem_rec, dim=0).to(self.device)

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
            spk, mem2 = self.readout(cur, mem2)
            # record spikes
            spk_rec.append(spk)
            mem_rec.append(mem2)

        return torch.stack(spk_rec, dim=0).to(self.device), torch.stack(mem_rec, dim=0).to(self.device)

    def _perform_stdp(self, spk_time):
        """

        :param spk_time: batch x total_size
        :return:
        """

        time_slice = spk_time[:, self.in_size:self.total_size]
        old_weight = self.fc1.weight
        stdp = self.stdp
        add_pre = -stdp.an * old_weight.T * (((spk_time == 0) * 1.0).T @ (torch.exp(-abs(time_slice) / stdp.tn)))
        add_post = stdp.ap * old_weight * (((time_slice == 0) * 1.0).T @ (torch.exp(-abs(spk_time) / stdp.tp)))
        ans = old_weight + add_pre.T + add_post
        stdp.count += torch.sum(spk_time == 0)
        # clamp
        cpos = torch.clamp(ans, stdp.wmin, stdp.wmax)
        cneg = torch.clamp(ans, -stdp.wmax, -stdp.wmin)
        self.fc1.weight.data = cpos * (ans > 0) + cneg * (ans < 0)

    def generate(self) -> List[SignalSource]:
        print(torch.sum(torch.floor(self.fc1.weight.data) != 0, 1).to(self.device),
              torch.sum(torch.floor(self.fc2.weight.data) != 0, 1).to(self.device))
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
                ws = round(self.fc1.weight.data[i][j].item())
                if ws != 0:
                    ans[self.in_size + i].add_conn(NeuronConnection(ans[j], ws))
        for i in range(self.out_size):
            for j in range(self.hidden_size):
                ws = round(self.fc2.weight.data[i][j].item())
                if ws != 0:
                    ans[self.in_size + self.hidden_size + i].add_conn(NeuronConnection(ans[self.in_size + j], ws))
        return ans
