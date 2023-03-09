import torch
import torch as th
from snntorch import surrogate

from data_processing.byte_loader import compress, decompress
from data_processing.dataloader import AudioMNIST
from data_processing.verilog_memory_generator import mem_gen
from modules.lsm_hyperparam import LSMInitParams, LSMNeuronParams, LSMInitializer, STDPLearner
from modules.lsm_net import LSMPool
from modules.verilog_generator import *
from trainers.trainer import OptmParams


def test_generate_hw3():
    n1 = NeuronDataInput()
    n2 = NeuronDataInput()
    n3 = NeuronDataInput()
    n4 = NeuronHidden(1, 8)
    n5 = NeuronHidden(1, 8)
    n6 = NeuronHidden(1, 8)
    n7 = NeuronReadout(1, 8)
    n8 = NeuronReadout(1, 8)
    n4.add_conn(NeuronConnection(n1, 3))
    n4.add_conn(NeuronConnection(n2, 3))
    n4.add_conn(NeuronConnection(n3, 2))
    n5.add_conn(NeuronConnection(n1, 1))
    n5.add_conn(NeuronConnection(n2, 2))
    n5.add_conn(NeuronConnection(n3, 3))
    n6.add_conn(NeuronConnection(n1, 4))
    n6.add_conn(NeuronConnection(n2, 3))
    n6.add_conn(NeuronConnection(n3, 4))
    n7.add_conn(NeuronConnection(n4, 3))
    n7.add_conn(NeuronConnection(n5, 2))
    n7.add_conn(NeuronConnection(n6, 3))
    n8.add_conn(NeuronConnection(n4, 2))
    n8.add_conn(NeuronConnection(n5, 4))
    n8.add_conn(NeuronConnection(n6, 2))
    neurons = [n1, n2, n3, n4, n5, n6, n7, n8]
    generate("./../verilog/wrapper.v", neurons)


def test_grad_compute():
    a: th.Tensor = th.tensor([1.], requires_grad=True)
    b: th.Tensor = a * a
    b.backward(gradient=th.tensor([1]))
    print(a.grad)


def test_lsm_gen():
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
    ans = net.generate()
    generate("./../verilog/generated.v", ans)


def save_file():
    a = (torch.rand((64, 32, 128)) > 0.5) * 1
    b = compress(a)
    torch.save(b.byte(), "./Datasets/test.pth")
    c = torch.load("./Datasets/test.pth")
    d = decompress(c)
    print(torch.sum(a != d))


def test_mem_gen():
    data = AudioMNIST("./SpikeData/")
    x, _ = data[0]
    mem_gen("./../verilog/memory_0.mem", x)


if __name__ == "__main__":
    test_mem_gen()
