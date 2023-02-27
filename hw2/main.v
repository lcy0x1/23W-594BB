`timescale 1ns/1ps
`define LEAK 1
`define BASE 6
`define THRESHOLD 8
`define W1 1
`define W2 2
`define W3 3

`define V_SIZE 3
`define I_SIZE 4
`define INF 5'b11111


/* 4-bit adder. 
 * valid input range: 0-15
 * If the result overflows, output will be 0b11111 meaning infinity
*/
module clipped_adder(
    input unsigned [`I_SIZE:0] a,
    input unsigned [`I_SIZE:0] b,
    output unsigned [`I_SIZE:0] out
);

wire unsigned [`I_SIZE:0] sum;
assign sum = a + b;
assign out = a[`I_SIZE] || b[`I_SIZE] || sum[`I_SIZE] ? `INF : sum;

endmodule

/* Central Arithmetic unit for LIF
 * out = base + x - y
 * base : 4-bit value
 * x    : 4-bit value, or 0b11111 for infinity
 * y    : 4-bit value
 *
 * if(x is infinity){
 *     return infinity;
 * } else {
 *     if(base + x > ey){
 *         if(base + x - ey is infinity){
 *             return infinity;
 *         } else {
 *             return base + x - ey;
 *         }
 *     } else {
 *         return 0;
 *     }
 * }
*/
module lif_core(
     input unsigned [`V_SIZE:0] prev_v,
     input unsigned [`I_SIZE:0] spike_in,
     input unsigned [`V_SIZE:0] leak,
     output unsigned [`I_SIZE:0] out
);

wire unsigned [`I_SIZE:0] padded_v = {1'b0, prev_v};
wire unsigned [`I_SIZE:0] padded_leak = {1'b0, leak};
wire unsigned [`I_SIZE:0] sum;
wire unsigned [`I_SIZE:0] ans;
assign sum = padded_v + spike_in;
assign ans = sum - padded_leak;

assign out = spike_in[`I_SIZE] ? `INF : sum > padded_leak ?  ans[`I_SIZE] ? `INF : ans : 0;

endmodule

module lif(
    input clk,
    input reset,
    input unsigned [`I_SIZE:0] spike_in,
    output reg spike_out,
    output unsigned [`V_SIZE:0] waveform
);

wire unsigned [`I_SIZE:0] sum;
wire unsigned [`V_SIZE:0] next_volt;
reg unsigned [`V_SIZE:0] voltage;
wire has_spike;
wire unsigned [`V_SIZE:0] leak = `LEAK;
lif_core add(voltage, spike_in, leak, sum);
assign has_spike = sum >= `THRESHOLD;
assign next_volt = has_spike ? 0 : sum;
assign waveform = voltage + `BASE;

always @(posedge clk) begin
    if (reset) begin
        voltage <= 0;
        spike_out <= 0;
    end else begin
        voltage <= next_volt;
        spike_out <= has_spike;
    end
end

endmodule

module wrapper(
    input clk,
    input reset,
    input s1,
    input s2,
    input s3,
    output spike,
    output unsigned [`V_SIZE:0] waveform
);

wire unsigned [`I_SIZE:0] i1 = s1 ? `W1 : 0;
wire unsigned [`I_SIZE:0] i2 = s2 ? `W2 : 0;
wire unsigned [`I_SIZE:0] i3 = s3 ? `W3 : 0;
wire unsigned [`I_SIZE:0] i12;
wire unsigned [`I_SIZE:0] i123;

clipped_adder a1(i1, i2, i12);
clipped_adder a2(i12, i3, i123);

lif ni(clk, reset, i123, spike, waveform);

endmodule