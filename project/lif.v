`timescale 1ns/1ps

`define DEF_LEAK 1
`define DEF_THRESHOLD 8
`define DEF_V_SIZE 4

`define INF {(V_SIZE+1){1'b1}}
`define V_WIRE unsigned [V_SIZE-1:0]
`define I_WIRE unsigned [V_SIZE:0]

/* clipped adder
 * input data should have width V_SIZE + 1
 * where the valid bits are V_SIZE, and there is a extra bit of 1 representing overflow
 */
module clipped_adder #(parameter V_SIZE = `DEF_V_SIZE) (
    input `I_WIRE a,
    input `I_WIRE b,
    output `I_WIRE out
);
wire `I_WIRE sum = a + b;
assign out = a[V_SIZE] || b[V_SIZE] || sum[V_SIZE] ? `INF : sum;
endmodule

module lif_core #(parameter V_SIZE = `DEF_V_SIZE) (
     input `V_WIRE prev_v,
     input `I_WIRE spike_in,
     input `V_WIRE leak,
     output `I_WIRE out
);

wire `I_WIRE padded_v = {1'b0, prev_v};
wire `I_WIRE padded_leak = {1'b0, leak};
wire `I_WIRE sum = padded_v + spike_in;
wire `I_WIRE ans = sum - padded_leak;
assign out = spike_in[V_SIZE] ? `INF : sum > padded_leak ? ans[V_SIZE] ? `INF : ans : 0;
endmodule

module lif #(
    parameter V_SIZE = `DEF_V_SIZE, 
    parameter THRESHOLD = `DEF_THRESHOLD, 
    parameter LEAK = `DEF_LEAK
    ) (
    input clk,
    input rstn,
    input `I_WIRE spike_in,
    output reg spike_out
);

wire `I_WIRE sum;
wire `V_WIRE next_volt = has_spike ? 0 : sum;
wire has_spike = sum >= THRESHOLD;
wire `V_WIRE leak = LEAK;

reg `V_WIRE voltage;

lif_core #(V_SIZE) add (voltage, spike_in, leak, sum);

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