import random

import numpy as np
import torch

import graph_util as gu


class LSMInitParams:

    def __init__(self, seed, fan_in, wlo, whi, inhib):
        self.seed = seed
        self.fan_in = fan_in
        self.wlo = wlo
        self.whi = whi
        self.inhib = inhib


class LSMInitializer:

    def __init__(self, in_size, hidden_size, out_size, param: LSMInitParams):
        self.in_size = in_size
        self.hidden_size = hidden_size
        self.out_size = out_size
        self.param = param
        pass

    def init_weight_generation(self, connect_array):
        """
        Generate weights with given connection map
        """
        with np.nditer(connect_array, op_flags=['readwrite']) as it:
            for x in it:
                x[...] = x * random.uniform(self.param.wlo, self.param.whi) 

        return connect_array

    def init_lsm_conn(self, fc):
        """
        Initialize connections.
        Be aware of the initial weights, sign of weights, number of inputs, and potential feedback loops.
        Number of inputs for each neuron should be at most fan_in

        :param fc: Linear(in_size + hidden_size, hidden_size), with weight size of (hidden_size, in_size + hidden_size)
        """
        np.random.seed(self.param.seed)
        connect_array = np.zeros((fc.weight.size(0), fc.weight.size(1)))
        input_neuron_size = connect_array.shape[1] - connect_array.shape[0]
        generated = False

        # Choose inhibitory neurons
        select_neuron = gu.select(connect_array.shape[0], self.param.inhib, 1, -1)
        neuron_list = [1] * input_neuron_size + select_neuron

        while not generated:
            # Graph Generation
            id = input_neuron_size-1 #Record which neuron is selected (in in_size + hidden_size)
            for i in connect_array.shape[0]:
                id += 1
                connection_selection = gu.select(connect_array.shape[1]-1, self.param.fan_in)
                connection_selection_padding = connection_selection[:id] + [0] + connection_selection[id:]
                connect_array[i, :] = np.multiply(connection_selection_padding, neuron_list)

            # Check the availability
            generated = gu.check_availability(connect_array)

        # Generate weights
        connect_array = self.init_weight_generation(connect_array)

        # Update weights to fc
        with torch.no_grad():
            fc.weight = torch.from_numpy(connect_array)

    def init_readout_weight(self, fc):
        """
        Initialize readout weights.

        :param fc: Linear(hidden_size, out_size), with weight size of (out_size, hidden_size)
        """
        pass  # TODO


class STDPLearner:

    def __init__(self):
        pass

    def stdp(self, weights_old, time_diff):
        """
        incrase weights for time_diff > 0, else reduce weights
        :param weights_old: previous weights
        :param time_diff: T_post - T_pre
        :return: updated weights (in batch)
        """
        return weights_old  # TODO
