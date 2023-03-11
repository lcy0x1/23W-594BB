import numpy as np
import torch

import modules.graph_util as gu


class LSMInitParams:

    def __init__(self, in_size, hidden_size, out_size, seed, fan_in, inhib):
        self.in_size = in_size
        self.hidden_size = hidden_size
        self.out_size = out_size
        self.seed = seed
        self.fan_in = fan_in
        self.inhib = inhib  # number of inhibitary neurons


class LSMNeuronParams:

    def __init__(self, wlo, whi, tlo, thi):
        self.wlo = wlo  # weight bound
        self.whi = whi  # weight bound
        self.tlo = tlo
        self.thi = thi


class LSMInitializer:

    def __init__(self, init: LSMInitParams, weights: LSMNeuronParams):
        #self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.device = torch.device("cpu")
        self.in_size = init.in_size
        self.hidden_size = init.hidden_size
        self.out_size = init.out_size
        self.param = init
        self.weights = weights
        #pass

    def init_lsm_conn(self, fc):
        """
        Initialize connections.
        Be aware of the initial weights, sign of weights, number of inputs, and potential feedback loops.
        Number of inputs for each neuron should be at most fan_in

        :param fc: Linear(in_size + hidden_size, hidden_size), with weight size of (hidden_size, in_size + hidden_size)
        """
        np.random.seed(self.param.seed)
        connect_array = fc.weight.data * 0
        generated = False

        # Choose inhibitory neurons
        select_neuron = gu.select(connect_array.size(0), self.param.inhib, 1, -1)
        neuron_list = [1] * self.in_size + select_neuron

        while not generated:
            # Graph Generation
            index = self.in_size - 1  # Record which neuron is selected (in in_size + hidden_size)
            for i in range(connect_array.size(0)):
                index += 1
                connection_selection = gu.select(connect_array.size(1) - 1, self.param.fan_in)
                connection_selection_padding = connection_selection[:index] + [0] + connection_selection[index:]
                connect_array[i, :] = torch.tensor(connection_selection_padding).to(self.device) * torch.tensor(neuron_list).to(self.device)

            # Check the availability
            generated = gu.check_availability(connect_array)

        # Generate weights
        connect_array *= torch.rand(connect_array.size()).to(self.device) * (self.weights.whi - self.weights.wlo) + self.weights.wlo

        # Update weights to fc
        with torch.no_grad():
            fc.weight.data = connect_array

    def init_readout_weight(self, fc):
        """
        Initialize readout weights.

        :param fc: Linear(hidden_size, out_size), with weight size of (out_size, hidden_size)
        """
        size = fc.weight.data.size()
        with torch.no_grad():
            fc.weight.data = torch.rand(size).to(self.device) * (self.weights.whi - self.weights.wlo) + self.weights.wlo

    def get_lsm_threshold(self):
        t0 = self.weights.tlo
        dt = self.weights.thi - t0
        return torch.rand(self.hidden_size) * dt + t0

    def get_readout_threshold(self):
        t0 = self.weights.tlo
        dt = self.weights.thi - t0
        return torch.rand(self.out_size) * dt + t0


class STDPLearner:

    def __init__(self, ap, an, tp, tn, wmax, wmin):
        """
        :param ap: A+ for STDP. Must be positive.
        :param an: A- for STDP. Must be positive.
        :param tp: tau+ fot STDP. Must be positive.
        :param tn: tau- fot STDP. Must be positive.
        :param wmax: max absolute value for weights, higher value capped. Must be positive.
        :param wmin: min absolute value for weights, lower value disconnects. Must be positive.
        """
        self.ap = ap
        self.an = an
        self.tp = tp
        self.tn = tn
        self.wmax = wmax
        self.wmin = wmin

        self.count = 0

    def step(self):
        ans = self.count
        self.count = 0
        return ans
