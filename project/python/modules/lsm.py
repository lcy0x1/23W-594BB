import torch
from torch import nn

from linear_leak import LinearLeakLIF


class LSMNeuron(LinearLeakLIF):

    def __init__(self, beta=1.0, threshold=1.0):
        super().__init__(beta, threshold, None, False, False)


class LSMPool(nn.Module):

    def __init__(self, size):
        super().__init__()
        # Initialize layers
        self.size = size
        self.fc1 = nn.Linear(size, size, bias=False)
        self.lif1 = LSMNeuron(threshold=10)
        self.init_weights()

    def forward(self, x):
        # Initialize hidden states at t=0
        mem = self.lif1.init_leaky()

        # Record the final layer
        spk_rec = []
        mem_rec = []

        spk = torch.zeros((self.size,))

        for step in range(self.optm_param.num_steps):
            cur1 = self.fc1(torch.concat((x[step], spk)))
            spk, mem = self.lif1(cur1, mem)
            spk_rec.append(spk)
            mem_rec.append(mem)

        return torch.stack(spk_rec, dim=0).detach(), torch.stack(mem_rec, dim=0)

    def init_weights(self):
        pass  # FIXME
