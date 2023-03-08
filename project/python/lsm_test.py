from snntorch import surrogate
from snntorch.functional import ce_rate_loss
from tqdm import tqdm

from data_processing.dataloader import DataParam, LoaderCreator
from modules.lsm_hyperparam import LSMNeuronParams
from modules.lsm_net import *
from trainers.spikegen import SpikeGenerator
from trainers.trainer import OptmParams, Trainer

DATAPATH = './Datasets/audio/'

train_param = DataParam(0.8, 64, shuffle=True)
val_param = DataParam(0.12, 64, shuffle=False)
test_param = DataParam(0.08, 32, shuffle=False)

train_dl, val_dl, test_dl = LoaderCreator(DATAPATH).create_loaders(
    train_param,
    val_param,
    test_param)

seed = 12345
weight_bit = 6
volt_bit = 8
weight_max = 2 ** weight_bit - 1
threshold_max = 2 ** volt_bit - 1
num_steps = 100
t_decay = 10
param = LSMInitParams(in_size=19, hidden_size=60, out_size=10, seed=seed, fan_in=16, inhib=10)
weights = LSMNeuronParams(wlo=4, whi=20, tlo=8, thi=threshold_max)
optm = OptmParams(grad=surrogate.fast_sigmoid(), num_steps=num_steps, lr=7e-4, beta_lo=1 - 1e-1, beta_hi=1 - 1e-3)
init = LSMInitializer(param, weights)
stdp = STDPLearner(ap=0.04, an=0.02, tp=t_decay, tn=t_decay, wmax=weight_max, wmin=0.5)
net = LSMPool(optm, param, init, stdp)

spikegen = SpikeGenerator(step=num_steps, scale=1, sp=4, sn=1, decay=0.9)
trainer = Trainer(net, optm, ce_rate_loss, spikegen.transform)

if __name__ == "__main__":
    w0 = net.fc1.weight.data.clone()
    for data, target in tqdm(iter(train_dl)):
        net.lsm_train(spikegen.transform(0, data))
        w1 = net.fc1.weight.data.clone()

        acvtivity = stdp.step() / 2
        variation = torch.sum(torch.square(w0 - w1))
        connected = torch.sum(w1 != 0)
        print(f"LSM activity: {acvtivity} | variation: {variation:.3f} | connected: {connected}")
        w0 = w1

# net.lsm_train()
# net.forward()
