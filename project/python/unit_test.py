import torch as th

from modules.verilog_generator import *


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


if __name__ == "__main__":
    test_generate_hw3()
