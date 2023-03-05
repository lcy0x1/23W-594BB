class LSMInitializer:

    def __init__(self, in_size, hidden_size, out_size, fan_in):
        self.in_size = in_size
        self.hidden_size = hidden_size
        self.out_size = out_size
        self.fan_in = fan_in
        pass

    def init_lsm_conn(self, fc):
        """
        Initialize connections.
        Be aware of the initial weights, sign of weights, number of inputs, and potential feedback loops.
        Number of inputs for each neuron should be at most fan_in

        :param fc: Linear(in_size + hidden_size, hidden_size), with weight size of (hidden_size, in_size + hidden_size)
        """
        pass  # TODO

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
