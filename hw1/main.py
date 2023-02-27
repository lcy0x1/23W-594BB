from lif import *


def hw1q1a():
    network = Network(dt=1.e-4, n=20000)
    node1 = LIF("n1", vref=-65, vth=-20, tau=0.05, r=1e6, spike=1)
    network.add_node(node1)
    source = CurrentSource(lambda t: 5.e-5 if 5000 <= t < 15000 else 0)
    network.sources.append(source)
    source.add_target(node1)
    network.simulate()
    return network


def hw1q1b():
    network = Network(dt=1.e-4, n=20000)
    node1 = LIF("n1", vref=-65, vth=-30, tau=0.05, r=1e6, spike=1)
    network.add_node(node1)
    source = CurrentSource(lambda t: 5.e-5 if 5000 <= t < 15000 else 0)
    network.sources.append(source)
    source.add_target(node1)
    network.simulate()
    return network


def hw1q2a():
    network = Network(dt=1.e-4, n=20000)

    node1 = LIF("n1", vref=-65, vth=-35, tau=0.05, r=1e6, spike=1)
    node2 = LIF("n2", vref=-65, vth=-35, tau=0.05, r=1e6, spike=1)
    node3 = LIF("n3", vref=-65, vth=-35, tau=0.05, r=1e6, spike=1)
    node4 = LIF("n4", vref=-65, vth=-60, tau=0.02, r=1e7, spike=1)
    node5 = LIF("n5", vref=-65, vth=-60, tau=0.02, r=1e7, spike=1)
    node1.add_target(node4, -1e-5)
    node2.add_target(node4, 1e-4)
    node3.add_target(node4, 1e-5)
    node1.add_target(node5, 3e-5)
    node2.add_target(node5, -4e-5)
    node3.add_target(node5, 5e-5)

    source = CurrentSource(lambda t: 5.e-5 if 5000 <= t < 15000 else 0)
    source.add_target(node1)
    source.add_target(node2)
    source.add_target(node3)

    network.add_node(node1)
    network.add_node(node2)
    network.add_node(node3)
    network.add_node(node4)
    network.add_node(node5)
    network.sources.append(source)
    network.simulate()
    return network


def hw1q2b():
    network = Network(dt=1.e-4, n=20000)

    node1 = LIF("n1", vref=-65, vth=-35, tau=0.05, r=1e6, spike=1)
    node2 = LIF("n2", vref=-65, vth=-35, tau=0.05, r=1e6, spike=1)
    node3 = LIF("n3", vref=-65, vth=-35, tau=0.05, r=1e6, spike=1)
    node4 = LIF("n4", vref=-65, vth=-60, tau=0.02, r=1e7, spike=1)
    node5 = LIF("n5", vref=-65, vth=-60, tau=0.02, r=1e7, spike=1)
    node1.add_target(node4, 1e-5)
    node2.add_target(node4, 1e-5)
    node3.add_target(node4, 1e-5)
    node1.add_target(node5, 5e-6)
    node2.add_target(node5, 5e-6)
    node3.add_target(node5, -1e-5)

    source = CurrentSource(lambda t: 5.e-5 if 5000 <= t < 15000 else 0)
    source.add_target(node1)
    source.add_target(node2)
    source.add_target(node3)

    network.add_node(node1)
    network.add_node(node2)
    network.add_node(node3)
    network.add_node(node4)
    network.add_node(node5)
    network.sources.append(source)
    network.simulate()
    return network


if __name__ == '__main__':
    print(hw1q2a().neurons["n5"].values[10000])
