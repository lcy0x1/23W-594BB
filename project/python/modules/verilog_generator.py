import math
from typing import List

"""
Author: Arthur Wang
Date: Mar 2
"""

TYPE_INPUT = "input"
TYPE_HIDDEN = "hidden"
TYPE_OUTPUT = "output"


class SignalSource:

    def __init__(self, neuron_type):
        self.neuron_type = neuron_type
        self.id = -1

    def produce(self, neuron_list: List, text: List[str]):
        self.id = len(neuron_list)
        neuron_list.append(self)

    def get_id(self):
        return f"{self.id:03d}"

    def get_spike(self):
        return f"spike_{self.id:03d}"

    def gen_module(self, text: List[str]):
        pass


class NeuronConnection:

    def __init__(self, in_neuron: SignalSource, weight):
        self.in_neuron = in_neuron
        self.weight = weight

    def get_weight(self):
        return self.weight


class NeuronDataInput(SignalSource):

    def __init__(self):
        super().__init__(TYPE_INPUT)

    def get_id(self):
        return f"{self.id:02d}i"

    def get_spike(self):
        return f"spikes_i[{self.id}]"


class NeuronDataImpl(SignalSource):

    def __init__(self, leak, threshold, neuron_type):
        super().__init__(neuron_type)
        self.leak = leak
        self.threshold = threshold
        self.fan_in: List[NeuronConnection] = []

    def add_conn(self, conn: NeuronConnection):
        self.fan_in.append(conn)

    def gen_module(self, text: List[str]):
        sid = self.get_id()
        zero = "{V_SIZE{1'b0}}"
        for conn in self.fan_in:
            pid = conn.in_neuron.get_id()
            text.append(f"wire `SIG_V x_{pid}_{sid} = {conn.in_neuron.get_spike()} ? {conn.get_weight()} : {zero};")
        summed = self.gen_add(text, 0, len(self.fan_in))
        text.append(f"lif #(V_SIZE,{self.threshold},{self.leak}) n{sid} "
                    f"(clk, rstn, {summed}, {self.get_spike()});")

    def gen_add(self, text, start, end):
        sid = self.get_id()
        idx = f"_{sid}_{start:03d}_{end - 1:03d}"
        if end - start == 2:
            aid = f"x_{self.fan_in[start].in_neuron.get_id()}_{sid}"
            bid = f"x_{self.fan_in[end - 1].in_neuron.get_id()}_{sid}"
        elif end - start == 3:
            aid = self.gen_add(text, start, end - 1)
            bid = f"x_{self.fan_in[end - 1].in_neuron.get_id()}_{sid}"
        else:
            mid = math.floor((start + end) / 2)
            aid = self.gen_add(text, start, mid)
            bid = self.gen_add(text, mid, end)
        text.append(f"wire `SIG_V sum{idx};")
        text.append(f"clipped_adder #(V_SIZE) add{idx}({aid}, {bid}, sum{idx});")
        return f"sum{idx}"


class NeuronHidden(NeuronDataImpl):

    def __init__(self, leak, threshold):
        super().__init__(leak, threshold, TYPE_HIDDEN)

    def gen_module(self, text: List[str]):
        text.append(f"wire {self.get_spike()};")
        super().gen_module(text)


class NeuronReadout(NeuronDataImpl):

    def __init__(self, leak, threshold):
        super().__init__(leak, threshold, TYPE_OUTPUT)

    def get_id(self):
        return f"{self.id:02d}o"

    def get_spike(self):
        return f"spikes_o[{self.id}]"


def generate(path: str, neurons: List[SignalSource]):
    input_count = sum([1 if i.neuron_type == TYPE_INPUT else 0 for i in neurons])
    output_count = sum([1 if i.neuron_type == TYPE_OUTPUT else 0 for i in neurons])
    ans_list = ['`include "lif.v"',
                "`timescale 1ns/1ps",
                "module generated #(parameter V_SIZE = `DEF_V_SIZE) (",
                "INDENT",
                "input wire clk,",
                "input wire rstn,",
                f"input wire [{input_count - 1}:0] spikes_i,",
                f"output wire [{output_count - 1}:0] spikes_o",
                "DEINDENT",
                ");"]
    in_list = []
    for n in neurons:
        if n.neuron_type == TYPE_INPUT:
            n.produce(in_list, ans_list)
    neuron_list = []
    for n in neurons:
        if n.neuron_type == TYPE_HIDDEN:
            n.produce(neuron_list, ans_list)
    for n in neurons:
        if n.neuron_type == TYPE_HIDDEN:
            n.gen_module(ans_list)
    out_list = []
    for n in neurons:
        if n.neuron_type == TYPE_OUTPUT:
            n.produce(out_list, ans_list)
            n.gen_module(ans_list)
    ans_list.append("endmodule")
    ans = ""
    indent = 0
    for line in ans_list:
        if line == "INDENT":
            indent += 1
        elif line == "DEINDENT":
            indent -= 1
        else:
            ans = ans + "\t" * indent + line + "\n"
    f = open(path, "w")
    f.write(ans)
    f.close()
