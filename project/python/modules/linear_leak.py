import snntorch as snn
import torch

"""
Author: Arthur Wang
Date: Mar 2
"""


class LinearLeakLIF(snn.Leaky):
    """
    This module use linear membrane potential decay instead of exponential one,
    and force membrane potential to be non-negative
    """

    def __init__(
            self,
            beta,
            threshold=1.0,
            spike_grad=None,
            learn_beta=False,
            learn_threshold=False
    ):
        super().__init__(
            beta=beta,
            threshold=threshold,
            spike_grad=spike_grad,
            init_hidden=False,
            inhibition=False,
            learn_beta=learn_beta,
            learn_threshold=learn_threshold,
            reset_mechanism="zero",
            state_quant=torch.relu,
            output=False
        )

    def _base_state_function(self, input_, mem):
        return mem - self.beta + input_

    def _base_state_function_hidden(self, input_):
        return self.mem - self.beta + input_
