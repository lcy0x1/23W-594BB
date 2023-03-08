`timescale 1ns/1ps

`define DEF_LEAK 1
`define DEF_THRESHOLD 8
`define DEF_V_SIZE 4

`define INF {(V_SIZE-1){1'b1}}
`define SIG_V signed [V_SIZE-1:0]
`define USG_V unsigned [V_SIZE-2:0]

module clipped_adder #(parameter V_SIZE = `DEF_V_SIZE) (
     input `SIG_V a,
     input `SIG_V b,
     output `SIG_V out
);

wire `SIG_V sum = a+b;
assign out = (!a[V_SIZE-1] && !b[V_SIZE-1] && sum[V_SIZE-1]) ? `INF : (a[V_SIZE-1] && b[V_SIZE-1] && !sum[V_SIZE-1]) ? {(V_SIZE){1'b1}} : sum;
endmodule

module lif_core #(parameter V_SIZE = `DEF_V_SIZE, parameter V_LEAK = `DEF_LEAK) (
     input `USG_V prev_v,
     input `SIG_V spike_in,
     output `USG_V out
);

wire `SIG_V padded_v = {1'b0, prev_v};
wire `SIG_V presum = padded_v + spike_in;
wire `SIG_V sum = presum - V_LEAK;
assign out = (!spike_in[V_SIZE-1] && presum[V_SIZE-1]) ? `INF : (presum[V_SIZE-1] || sum[V_SIZE-1]) ? 0 : sum[V_SIZE-2:0];
endmodule

module lif #(
    parameter V_SIZE = `DEF_V_SIZE, 
    parameter THRESHOLD = `DEF_THRESHOLD, 
    parameter V_LEAK = `DEF_LEAK
    ) (
    input clk,
    input rstn,
    input `SIG_V spike_in,
    output reg spike_out
);

wire `USG_V sum;
wire has_spike = sum >= THRESHOLD;
wire `USG_V next_volt = has_spike ? 0 : sum;

reg `USG_V voltage;

lif_core #(V_SIZE, V_LEAK) add (voltage, spike_in, sum);

always @(posedge clk) begin
    if (!rstn) begin
        voltage <= 0;
        spike_out <= 0;
    end else begin
        voltage <= next_volt;
        spike_out <= has_spike;
    end
end

endmodule