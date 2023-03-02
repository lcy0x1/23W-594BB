from generator import *


def test():
    n1 = NeuronDataInput()
    n2 = NeuronDataInput()
    n3 = NeuronDataInput()
    n4 = NeuronDataImpl(1, 8, TYPE_EXCITATORY)
    n5 = NeuronDataImpl(1, 8, TYPE_EXCITATORY)
    n6 = NeuronDataImpl(1, 8, TYPE_EXCITATORY)
    n7 = NeuronDataImpl(1, 8, TYPE_EXCITATORY)
    n8 = NeuronDataImpl(1, 8, TYPE_EXCITATORY)
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

if __name__ == "__main__":
    test()
