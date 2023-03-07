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
        summed = ""
        zero = "{V_SIZE{1'b0}}"
        for conn in self.fan_in:
            pid = conn.in_neuron.get_id()
            text.append(f"parameter `SIG_V w_{pid}_{sid} = {conn.get_weight()};")
            if len(summed) > 0:
                summed += " + "
            summed += f"({conn.in_neuron.get_spike()} ? w_{pid}_{sid} : {zero})"

        text.append(f"lif #(V_SIZE,{self.threshold},{self.leak}) n{sid} "
                    f"(clk, rstn, {summed}, {self.get_spike()});")


class NeuronHidden(NeuronDataImpl):

    def __init__(self, leak, threshold):
        super().__init__(leak, threshold, TYPE_HIDDEN)

    def produce(self, neuron_list: List, text: List[str]):
        super().produce(neuron_list, text)
        text.append(f"wire {self.get_spike()};")
        self.gen_module(text)


class NeuronReadout(NeuronDataImpl):

    def __init__(self, leak, threshold):
        super().__init__(leak, threshold, TYPE_OUTPUT)

    def get_id(self):
        return f"{self.id:02d}o"

    def get_spike(self):
        return f"spikes_o[{self.id}]"

    def produce(self, neuron_list: List, text: List[str]):
        super().produce(neuron_list, text)
        self.gen_module(text)


def generate(path: str, neurons: List[SignalSource]):
    ans_list = []
    ans_list.append('`include "lif.v"')
    ans_list.append("module wrapper #(parameter V_SIZE = `DEF_V_SIZE) (")
    ans_list.append("INDENT")
    ans_list.append("input wire clk,")
    ans_list.append("input wire rstn,")
    input_count = sum([1 if i.neuron_type == TYPE_INPUT else 0 for i in neurons])
    output_count = sum([1 if i.neuron_type == TYPE_OUTPUT else 0 for i in neurons])
    ans_list.append(f"input wire [{input_count - 1}:0] spikes_i")
    ans_list.append(f"output wire [{output_count - 1}:0] spikes_o")
    ans_list.append("DEINDENT")
    ans_list.append(");")
    in_list = []
    for n in neurons:
        if n.neuron_type == TYPE_INPUT:
            n.produce(in_list, ans_list)
    neuron_list = []
    for n in neurons:
        if n.neuron_type == TYPE_HIDDEN:
            n.produce(neuron_list, ans_list)
    out_list = []
    for n in neurons:
        if n.neuron_type == TYPE_OUTPUT:
            n.produce(out_list, ans_list)
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
