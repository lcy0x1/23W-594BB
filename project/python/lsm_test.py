from snntorch import surrogate
from snntorch.functional import ce_rate_loss

from modules.lsm_net import *
from trainers.spikegen import SpikeGenerator
from trainers.trainer import OptmParams, Trainer

seed = 12345
weight_bit = 6
weight_max = 2 ** weight_bit - 1
num_steps = 100
t_decay = 10
param = LSMInitParams(in_size=19, hidden_size=60, out_size=10, seed=seed, fan_in=16, wlo=4, whi=20, inhib=10)
optm = OptmParams(grad=surrogate.fast_sigmoid(), num_steps=num_steps, lr=7e-4, beta_lo=1 - 1e-1, beta_hi=1 - 1e-3)
init = LSMInitializer(param)
stdp = STDPLearner(ap=0.04, an=0.02, tp=t_decay, tn=t_decay, wmax=weight_max, wmin=0.5)
net = LSMPool(optm, param, init, stdp)

spikegen = SpikeGenerator(step=num_steps, scale=1, sp=4, sn=1, decay=0.9)
trainer = Trainer(net, optm, ce_rate_loss, spikegen.transform)

# net.lsm_train()
# net.forward()
