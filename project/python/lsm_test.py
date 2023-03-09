import os

from snntorch import surrogate
from snntorch.functional import ce_rate_loss
from tqdm import tqdm

from data_processing.dataloader import DataParam, LoaderCreator
from modules.lsm_hyperparam import LSMNeuronParams
from modules.lsm_net import *
from trainers.trainer import OptmParams, Trainer


def build_env():
    train_param = DataParam(0.8, 64, shuffle=True)
    val_param = DataParam(0.12, 64, shuffle=False)
    test_param = DataParam(0.08, 32, shuffle=False)

    train_dl, val_dl, test_dl = LoaderCreator('./SpikeData/').create_loaders(
        train_param,
        val_param,
        test_param)

    seed = 12345
    weight_bit = 6
    volt_bit = 8
    weight_max = 2 ** weight_bit - 1
    threshold_max = 2 ** volt_bit - 1
    num_steps = 128
    t_decay = 10
    param = LSMInitParams(in_size=19, hidden_size=60, out_size=10, seed=seed, fan_in=16, inhib=10)
    weights = LSMNeuronParams(wlo=4, whi=20, tlo=8, thi=threshold_max)
    optm = OptmParams(grad=surrogate.fast_sigmoid(), num_steps=num_steps, lr=7e-4, beta_lo=1 - 1e-1, beta_hi=1 - 1e-3)
    init = LSMInitializer(param, weights)
    stdp = STDPLearner(ap=0.04, an=0.02, tp=t_decay, tn=t_decay, wmax=weight_max, wmin=0.5)
    net = LSMPool(optm, param, init, stdp)

    def transform(_, x):
        return x.permute((2, 0, 1))

    trainer = Trainer(net, optm, ce_rate_loss, transform)

    w0 = net.fc1.weight.data.clone()

    checkpoint = "./Checkpoints"
    if not os.path.isdir(checkpoint):
        os.mkdir(checkpoint)
    epochs = tqdm(iter(train_dl))
    for i, (data, target) in enumerate(epochs):
        net.lsm_train(transform(0, data))
        w1 = net.fc1.weight.data.clone()

        acvtivity = stdp.step() / 2
        variation = torch.sum(torch.square(w0 - w1))
        connected = torch.sum(w1 != 0)
        epochs.set_description(f"LSM activity: {acvtivity} | variation: {variation:.3f} | connected: {connected}")
        w0 = w1

        torch.save(net.state_dict(), f"{checkpoint}/lsm_stdp_{i}.pth")

    trainer.train(100, train_dl, test_dl)

    # net.lsm_train()
    # net.forward()


if __name__ == "__main__":
    build_env()
