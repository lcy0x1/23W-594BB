`include "lif.v"
module wrapper #(parameter V_SIZE = `DEF_V_SIZE) (
	input wire clk,
	input wire rstn,
	input wire [2:0] spikes_i
	output wire [1:0] spikes_o
);
wire spike_000;
parameter `SIG_V w_00i_000 = 3;
parameter `SIG_V w_01i_000 = 3;
parameter `SIG_V w_02i_000 = 2;
lif #(V_SIZE,8,1) n000 (clk, rstn, (spikes_i[0] ? w_00i_000 : {V_SIZE{1'b0}}) + (spikes_i[1] ? w_01i_000 : {V_SIZE{1'b0}}) + (spikes_i[2] ? w_02i_000 : {V_SIZE{1'b0}}), spike_000);
wire spike_001;
parameter `SIG_V w_00i_001 = 1;
parameter `SIG_V w_01i_001 = 2;
parameter `SIG_V w_02i_001 = 3;
lif #(V_SIZE,8,1) n001 (clk, rstn, (spikes_i[0] ? w_00i_001 : {V_SIZE{1'b0}}) + (spikes_i[1] ? w_01i_001 : {V_SIZE{1'b0}}) + (spikes_i[2] ? w_02i_001 : {V_SIZE{1'b0}}), spike_001);
wire spike_002;
parameter `SIG_V w_00i_002 = 4;
parameter `SIG_V w_01i_002 = 3;
parameter `SIG_V w_02i_002 = 4;
lif #(V_SIZE,8,1) n002 (clk, rstn, (spikes_i[0] ? w_00i_002 : {V_SIZE{1'b0}}) + (spikes_i[1] ? w_01i_002 : {V_SIZE{1'b0}}) + (spikes_i[2] ? w_02i_002 : {V_SIZE{1'b0}}), spike_002);
parameter `SIG_V w_000_00o = 3;
parameter `SIG_V w_001_00o = 2;
parameter `SIG_V w_002_00o = 3;
lif #(V_SIZE,8,1) n00o (clk, rstn, (spike_000 ? w_000_00o : {V_SIZE{1'b0}}) + (spike_001 ? w_001_00o : {V_SIZE{1'b0}}) + (spike_002 ? w_002_00o : {V_SIZE{1'b0}}), spikes_o[0]);
parameter `SIG_V w_000_01o = 2;
parameter `SIG_V w_001_01o = 4;
parameter `SIG_V w_002_01o = 2;
lif #(V_SIZE,8,1) n01o (clk, rstn, (spike_000 ? w_000_01o : {V_SIZE{1'b0}}) + (spike_001 ? w_001_01o : {V_SIZE{1'b0}}) + (spike_002 ? w_002_01o : {V_SIZE{1'b0}}), spikes_o[1]);
endmodule
