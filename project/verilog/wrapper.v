`include "lif.v"
module wrapper #(parameter V_SIZE = `DEF_V_SIZE) (
	input wire clk,
	input wire rstn,
	input wire spike_001,
	input wire spike_002,
	input wire spike_003
);
wire spike_004;
parameter `SIG_V w_001_004 = -3;
parameter `SIG_V w_002_004 = -3;
parameter `SIG_V w_003_004 = -2;
lif #(V_SIZE,8,1) n004 (clk, rstn, (spike_001 ? w_001_004 : {V_SIZE{1'b0}}) + (spike_002 ? w_002_004 : {V_SIZE{1'b0}}) + (spike_003 ? w_003_004 : {V_SIZE{1'b0}}), spike_004);
wire spike_005;
parameter `SIG_V w_001_005 = -1;
parameter `SIG_V w_002_005 = -2;
parameter `SIG_V w_003_005 = -3;
lif #(V_SIZE,8,1) n005 (clk, rstn, (spike_001 ? w_001_005 : {V_SIZE{1'b0}}) + (spike_002 ? w_002_005 : {V_SIZE{1'b0}}) + (spike_003 ? w_003_005 : {V_SIZE{1'b0}}), spike_005);
wire spike_006;
parameter `SIG_V w_001_006 = -4;
parameter `SIG_V w_002_006 = -3;
parameter `SIG_V w_003_006 = -4;
lif #(V_SIZE,8,1) n006 (clk, rstn, (spike_001 ? w_001_006 : {V_SIZE{1'b0}}) + (spike_002 ? w_002_006 : {V_SIZE{1'b0}}) + (spike_003 ? w_003_006 : {V_SIZE{1'b0}}), spike_006);
wire spike_007;
parameter `SIG_V w_004_007 = 3;
parameter `SIG_V w_005_007 = 2;
parameter `SIG_V w_006_007 = 3;
lif #(V_SIZE,8,1) n007 (clk, rstn, (spike_004 ? w_004_007 : {V_SIZE{1'b0}}) + (spike_005 ? w_005_007 : {V_SIZE{1'b0}}) + (spike_006 ? w_006_007 : {V_SIZE{1'b0}}), spike_007);
wire spike_008;
parameter `SIG_V w_004_008 = 2;
parameter `SIG_V w_005_008 = 4;
parameter `SIG_V w_006_008 = 2;
lif #(V_SIZE,8,1) n008 (clk, rstn, (spike_004 ? w_004_008 : {V_SIZE{1'b0}}) + (spike_005 ? w_005_008 : {V_SIZE{1'b0}}) + (spike_006 ? w_006_008 : {V_SIZE{1'b0}}), spike_008);
endmodule
