import torch
from torch import nn
import numpy as np
import random
import graph_util as gu
class LSMInitializer:

    def __init__(self, in_size, hidden_size, out_size, fan_in):
        self.in_size = in_size
        self.hidden_size = hidden_size
        self.out_size = out_size
        self.fan_in = fan_in
        pass

    def init_weight_generation(self, connect_array, weights_LB = 1, weights_UB = 2):
        """
        Generate weights with given connection map
        """
        with np.nditer(connect_array, op_flags=['readwrite']) as it:
            for x in it:
                if x > 0:
                    x = random.uniform(weights_LB,weights_UB)
        
        return connect_array

    def init_lsm_conn(self, fc, weights_LB = 1, weights_UB = 2):
        """
        Initialize connections.
        Be aware of the initial weights, sign of weights, number of inputs, and potential feedback loops.
        Number of inputs for each neuron should be at most fan_in

        :param fc: Linear(in_size + hidden_size, hidden_size), with weight size of (hidden_size, in_size + hidden_size)
        """
        np.random.seed(114514)
        connect_array = np.zeros(list(fc.weight.shape)[0],list(fc.weight.shape)[1])
        generated = False

        while not generated:
            # Graph Generation
            for i in connect_array.shape[0]:
                connect_array[i,:] = gu.select(connect_array.shape[1],16)

            #Generate weights
            connect_array = self.init_weight_generation(connect_array)

            #Check the availability
            generated = gu.check_availability(connect_array)

        #Update weights to fc
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
