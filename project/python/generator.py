from typing import List

TYPE_INPUT = "input"
TYPE_EXCITATORY = "excitatory"
TYPE_INHIBITORY = "inhibitory"


class SignalSource:

    def __init__(self, neuron_type):
        self.neuron_type = neuron_type
        self.id = -1

    def produce(self, neuron_list: List, text: List[str]):
        neuron_list.append(self)
        self.id = len(neuron_list)


class NeuronConnection:

    def __init__(self, in_neuron: SignalSource, weight):
        self.in_neuron = in_neuron
        self.weight = weight

    def get_weight(self):
        return self.weight if self.in_neuron.neuron_type == TYPE_EXCITATORY else -self.weight


class NeuronDataInput(SignalSource):

    def __init__(self):
        super().__init__(TYPE_INPUT)

    def produce(self, neuron_list: List, text: List[str]):
        super().produce(neuron_list, text)
        sid = f"{self.id:03d}"
        if len(text) > 0:
            text[-1] += ","
        text.append(f"input wire spike_{sid}")


class NeuronDataImpl(SignalSource):

    def __init__(self, leak, threshold, neuron_type):
        super().__init__(neuron_type)
        self.leak = leak
        self.threshold = threshold
        self.fan_in: List[NeuronConnection] = []

    def add_conn(self, conn: NeuronConnection):
        self.fan_in.append(conn)

    def produce(self, neuron_list: List, text: List[str]):
        super().produce(neuron_list, text)
        sid = f"{self.id:03d}"
        text.append(f"wire spike_{sid};")
        summed_signals = ""
        zero = "{V_SIZE{1'b0}}"
        for conn in self.fan_in:
            pid = f"{conn.in_neuron.id:03d}"
            text.append(f"parameter `SIG_V w_{pid}_{sid} = {conn.get_weight()};")
            if len(summed_signals) > 0:
                summed_signals += " + "
            summed_signals += f"(spike_{pid} ? w_{pid}_{sid} : {zero})"

        text.append(f"lif #(V_SIZE,{self.threshold},{self.leak}) n{sid} (clk, rstn, {summed_signals}, spike_{sid});")


def generate(path: str, neurons: List[SignalSource]):
    neuron_list = []
    ans_list = []
    ans_list.append('`include "lif.v"')
    #ans_list.append('`define SIG_V signed [V_SIZE-1:0]')
    ans_list.append("module wrapper #(parameter V_SIZE = `DEF_V_SIZE) (")
    ans_list.append("INDENT")
    ans_list.append("input wire clk,")
    ans_list.append("input wire rstn,")
    input_text = []
    for n in neurons:
        if n.neuron_type == TYPE_INPUT:
            n.produce(neuron_list, input_text)
    ans_list.extend(input_text)
    ans_list.append("DEINDENT")
    ans_list.append(");")
    for n in neurons:
        if n.neuron_type != TYPE_INPUT:
            n.produce(neuron_list, ans_list)
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
