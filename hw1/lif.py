from typing import final, List, Dict


class Target:

    def __init__(self, target, w: float):
        self.target: final = target
        self.w: final = w

    def fire(self, dt, voltage):
        self.target.add_charge(self.w * voltage * dt)


class LIF:

    def __init__(self, name: str, vref: float, vth: float, tau: float, r: float, spike: float):
        self.name: final = name
        self.vref: final = vref
        self.vth: final = vth
        self.tau: final = tau
        self.r: final = r
        self.spike: final = spike

        self.targets: List[Target] = []
        self.v = vref
        self.fire = 0

        self.indexes = []
        self.values = []
        self.spikes = []

    def add_target(self, target, w: float):
        self.targets.append(Target(target, w))

    def add_charge(self, charge: float):
        self.v += charge * self.r / self.tau

    def tick(self, dt):
        if self.v >= self.vth:
            for t in self.targets:
                t.fire(dt, self.spike)
            self.fire += 1
            self.v = self.vref
            self.add_record(dt, self.vth)
            self.add_record(0, self.vref)
        else:
            self.v -= (self.v - self.vref) / self.tau * dt
            self.add_record(dt, self.v)

    def add_record(self, dt, v):
        old = 0 if len(self.indexes) == 0 else self.indexes[-1]
        self.indexes.append(old + dt)
        self.values.append(v)
        if dt == 0:
            self.spikes.append(old)

    def plot(self, plt):
        plt.plot(self.indexes, self.values, '-')
        plt.plot(self.spikes, self.spikes * 0 + [self.vth] * self.fire, 'o')
        plt.plot([self.indexes[0], self.indexes[-1]], [self.vth, self.vth], '--')
        plt.legend(["neuron voltage", "output spikes", "threshold"], loc="lower right")
        plt.title(f'{self.name} fires {self.fire} times')


class CurrentSource:

    def __init__(self, func):
        self.func = func
        self.targets: List[LIF] = []

    def add_target(self, target: LIF):
        self.targets.append(target)

    def tick(self, dt, time):
        val = self.func(time)
        for t in self.targets:
            t.add_charge(dt * val)


class Network:

    def __init__(self, dt: float, n: int):
        self.n: final = n
        self.dt: final = dt
        self.sources: List[CurrentSource] = []
        self.neurons: Dict[str, LIF] = {}

        self.time = 0

    def add_node(self, node: LIF):
        self.neurons[node.name] = node

    def tick(self):
        for s in self.sources:
            s.tick(self.dt, self.time)
        for v in self.neurons:
            self.neurons[v].tick(self.dt)
        self.time += 1

    def simulate(self):
        for i in range(self.n):
            self.tick()
