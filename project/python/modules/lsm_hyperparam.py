import random

import numpy as np
import torch

import graph_util as gu


class LSMInitParams:

    def __init__(self, seed, fan_in, wlo, whi):
        self.seed = seed
        self.fan_in = fan_in
        self.wlo = wlo
        self.whi = whi


class LSMInitializer:

    def __init__(self, in_size, hidden_size, out_size, fan_in):
        self.in_size = in_size
        self.hidden_size = hidden_size
        self.out_size = out_size
        self.fan_in = fan_in
        pass

    def init_weight_generation(self, connect_array, weights_LB=1, weights_UB=2):
        """
        Generate weights with given connection map
        """
        with np.nditer(connect_array, op_flags=['readwrite']) as it:
            for x in it:
                x = x * random.uniform(weights_LB, weights_UB)

        return connect_array

    def init_lsm_conn(self, fc, weights_LB=1, weights_UB=2, inhibitory_num=8):
        """
        Initialize connections.
        Be aware of the initial weights, sign of weights, number of inputs, and potential feedback loops.
        Number of inputs for each neuron should be at most fan_in

        :param fc: Linear(in_size + hidden_size, hidden_size), with weight size of (hidden_size, in_size + hidden_size)
        """
        np.random.seed(114514)
        connect_array = np.zeros(list(fc.weight.shape)[0], list(fc.weight.shape)[1])
        generated = False

        # Choose inhibitory neurons
        select_neuron = gu.select(connect_array.shape[0], inhibitory_num, 1, -1)
        neuron_list = [1] * (connect_array.shape[1] - connect_array.shape[0]) + select_neuron

        while not generated:
            # Graph Generation
            for i in connect_array.shape[0]:
                connect_array[i, :] = np.multiply(gu.select(connect_array.shape[1], 16), neuron_list)

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
