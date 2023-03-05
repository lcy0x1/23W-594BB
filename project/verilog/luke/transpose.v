/*

Author: Jiachen Zhang, Arthur Wang
Creation Date: Nov 23 
Last Modified: Nov 29

module trans: similar to mac, but for transpose
module t8x8: 8 by 8 matrix transpose module

->  clk: clock
->  enable: global enable, nothing shall be done if it is low
->  reset: reset everything
->  do_transpose: enable for transpose. If low, output delayed input
->  x_in: input to be transposed
->  en: signals for the duration of valid data
->  z_out: the output, transposed or not
->  in_mult_clear: the clear_out flag provided by the memory
->  out_mult_clear: the clear_in flag required by the multiplier

*/

module trans(
    input clk,
    input enable,
    input reset,
    input [32:0] x_in,
    input [31:0] y_in,
    input v_in,
    input clear_in,
    input shift,
    output reg [32:0] x_out,
    output reg [31:0] y_out,
    output reg v_out,
    output reg clear_out,
    input x_shift,
    input [32:0] x_delay_in,
    output reg [32:0] x_delay_out
);
    reg [32:0] xr;
    reg [32:0] standby;

    always @(posedge clk) begin
        if(reset) begin
            x_out <= 0;
            y_out <= 0;
            v_out <= 0;
            xr <= 0;
            clear_out <= 0;
            standby <= 0;
            x_delay_out <= 0;
        end else if(enable) begin
            x_out <= x_in;
            y_out <= shift ? standby[31:0] : y_in;
            x_delay_out <= x_shift ? standby : x_delay_in;
            v_out <= v_in;
            xr <= v_in ? x_in : xr;
            clear_out <= clear_in;
            standby <= clear_in ? xr : standby;
        end
    end
endmodule

module t8x8(
    input clk,
    input enable,
    input reset,
    input do_transpose,
    input [31:0] x_in [7:0],
    input en,
    output [31:0] z_out [7:0],
    input [7:0] in_mult_clear,
    output reg [7:0] out_mult_clear
);

wire [32:0] x_mid [7:0][6:0];
wire [31:0] y_mid [6:0][7:0];
wire v_mid [6:0][7:0];
wire clear_mid [7:0][7:0];
wire [32:0] x_delay_mid [7:0][6:0];
reg [7:0] v_in;
reg [7:0] clear_in;
reg [2:0] v_count;
reg en_delay;

wire [31:0] y_out [7:0];
wire [32:0] x_delay_out [7:0];
wire [7:0] clear_out;
wire [7:0] v_out;
wire [32:0] x_out [7:0];

wire [7:0] out_mult_clear_pre;

wire do_clock;
assign do_clock = en || en_delay || v_count > 0;

always @(posedge clk) begin
    if(reset)begin
        v_count <= 0;
        v_in <= 0;
        clear_in <= 0;
        en_delay <= 0;
    end
    else if(enable)begin
        en_delay <= en ? 1 : 0;
        v_in <= v_count[0] == 0 ? 8'h11 << v_count[2:1] : 0;
        if(do_clock) begin
            v_count <= v_count == 7 ? 0 : v_count + 1;
        end 
        else begin
            v_count <= 0;
        end
        clear_in <= do_clock ? 1 << v_count : 0;
        out_mult_clear <= out_mult_clear_pre;
    end
end

assign z_out[0] = do_transpose ? y_out[0] : x_delay_out[0][31:0];
assign z_out[1] = do_transpose ? y_out[1] : x_delay_out[1][31:0];
assign z_out[2] = do_transpose ? y_out[2] : x_delay_out[2][31:0];
assign z_out[3] = do_transpose ? y_out[3] : x_delay_out[3][31:0];
assign z_out[4] = do_transpose ? y_out[4] : x_delay_out[4][31:0];
assign z_out[5] = do_transpose ? y_out[5] : x_delay_out[5][31:0];
assign z_out[6] = do_transpose ? y_out[6] : x_delay_out[6][31:0];
assign z_out[7] = do_transpose ? y_out[7] : x_delay_out[7][31:0];

assign out_mult_clear_pre[0] = x_delay_out[0][32];
assign out_mult_clear_pre[1] = x_delay_out[1][32];
assign out_mult_clear_pre[2] = x_delay_out[2][32];
assign out_mult_clear_pre[3] = x_delay_out[3][32];
assign out_mult_clear_pre[4] = x_delay_out[4][32];
assign out_mult_clear_pre[5] = x_delay_out[5][32];
assign out_mult_clear_pre[6] = x_delay_out[6][32];
assign out_mult_clear_pre[7] = x_delay_out[7][32];



trans t00(clk, enable, reset, {in_mult_clear[0], x_in[0]}, y_mid[0][0],     v_in[0], clear_in[0], clear_out[0], x_mid[0][0],    y_out[0], v_mid[0][0], clear_mid[0][0], clear_out[0], x_delay_mid[0][0], x_delay_out[0]);
trans t10(clk, enable, reset, {in_mult_clear[1], x_in[1]}, y_mid[1][0], v_mid[0][0], clear_in[1], clear_out[0], x_mid[1][0], y_mid[0][0], v_mid[1][0], clear_mid[1][0], clear_out[1], x_delay_mid[1][0], x_delay_out[1]);
trans t20(clk, enable, reset, {in_mult_clear[2], x_in[2]}, y_mid[2][0], v_mid[1][0], clear_in[2], clear_out[0], x_mid[2][0], y_mid[1][0], v_mid[2][0], clear_mid[2][0], clear_out[2], x_delay_mid[2][0], x_delay_out[2]);
trans t30(clk, enable, reset, {in_mult_clear[3], x_in[3]}, y_mid[3][0], v_mid[2][0], clear_in[3], clear_out[0], x_mid[3][0], y_mid[2][0], v_mid[3][0], clear_mid[3][0], clear_out[3], x_delay_mid[3][0], x_delay_out[3]);
trans t40(clk, enable, reset, {in_mult_clear[4], x_in[4]}, y_mid[4][0], v_mid[3][0], clear_in[4], clear_out[0], x_mid[4][0], y_mid[3][0], v_mid[4][0], clear_mid[4][0], clear_out[4], x_delay_mid[4][0], x_delay_out[4]);
trans t50(clk, enable, reset, {in_mult_clear[5], x_in[5]}, y_mid[5][0], v_mid[4][0], clear_in[5], clear_out[0], x_mid[5][0], y_mid[4][0], v_mid[5][0], clear_mid[5][0], clear_out[5], x_delay_mid[5][0], x_delay_out[5]);
trans t60(clk, enable, reset, {in_mult_clear[6], x_in[6]}, y_mid[6][0], v_mid[5][0], clear_in[6], clear_out[0], x_mid[6][0], y_mid[5][0], v_mid[6][0], clear_mid[6][0], clear_out[6], x_delay_mid[6][0], x_delay_out[6]);
trans t70(clk, enable, reset, {in_mult_clear[7], x_in[7]}, 32'b0,       v_mid[6][0], clear_in[7], clear_out[0], x_mid[7][0], y_mid[6][0],    v_out[0], clear_mid[7][0], clear_out[7], x_delay_mid[7][0], x_delay_out[7]);

trans t01(clk, enable, reset, x_mid[0][0], y_mid[0][1],     v_in[1], clear_mid[0][0], clear_out[1], x_mid[0][1],    y_out[1], v_mid[0][1], clear_mid[0][1], clear_out[0], x_delay_mid[0][1], x_delay_mid[0][0]);
trans t11(clk, enable, reset, x_mid[1][0], y_mid[1][1], v_mid[0][1], clear_mid[1][0], clear_out[1], x_mid[1][1], y_mid[0][1], v_mid[1][1], clear_mid[1][1], clear_out[1], x_delay_mid[1][1], x_delay_mid[1][0]);
trans t21(clk, enable, reset, x_mid[2][0], y_mid[2][1], v_mid[1][1], clear_mid[2][0], clear_out[1], x_mid[2][1], y_mid[1][1], v_mid[2][1], clear_mid[2][1], clear_out[2], x_delay_mid[2][1], x_delay_mid[2][0]);
trans t31(clk, enable, reset, x_mid[3][0], y_mid[3][1], v_mid[2][1], clear_mid[3][0], clear_out[1], x_mid[3][1], y_mid[2][1], v_mid[3][1], clear_mid[3][1], clear_out[3], x_delay_mid[3][1], x_delay_mid[3][0]);
trans t41(clk, enable, reset, x_mid[4][0], y_mid[4][1], v_mid[3][1], clear_mid[4][0], clear_out[1], x_mid[4][1], y_mid[3][1], v_mid[4][1], clear_mid[4][1], clear_out[4], x_delay_mid[4][1], x_delay_mid[4][0]);
trans t51(clk, enable, reset, x_mid[5][0], y_mid[5][1], v_mid[4][1], clear_mid[5][0], clear_out[1], x_mid[5][1], y_mid[4][1], v_mid[5][1], clear_mid[5][1], clear_out[5], x_delay_mid[5][1], x_delay_mid[5][0]);
trans t61(clk, enable, reset, x_mid[6][0], y_mid[6][1], v_mid[5][1], clear_mid[6][0], clear_out[1], x_mid[6][1], y_mid[5][1], v_mid[6][1], clear_mid[6][1], clear_out[6], x_delay_mid[6][1], x_delay_mid[6][0]);
trans t71(clk, enable, reset, x_mid[7][0],     32'b0, v_mid[6][1],   clear_mid[7][0], clear_out[1], x_mid[7][1], y_mid[6][1],    v_out[1], clear_mid[7][1], clear_out[7], x_delay_mid[7][1], x_delay_mid[7][0]);

trans t02(clk, enable, reset, x_mid[0][1], y_mid[0][2],     v_in[2], clear_mid[0][1], clear_out[2], x_mid[0][2],    y_out[2], v_mid[0][2], clear_mid[0][2], clear_out[0], x_delay_mid[0][2], x_delay_mid[0][1]);
trans t12(clk, enable, reset, x_mid[1][1], y_mid[1][2], v_mid[0][2], clear_mid[1][1], clear_out[2], x_mid[1][2], y_mid[0][2], v_mid[1][2], clear_mid[1][2], clear_out[1], x_delay_mid[1][2], x_delay_mid[1][1]);
trans t22(clk, enable, reset, x_mid[2][1], y_mid[2][2], v_mid[1][2], clear_mid[2][1], clear_out[2], x_mid[2][2], y_mid[1][2], v_mid[2][2], clear_mid[2][2], clear_out[2], x_delay_mid[2][2], x_delay_mid[2][1]);
trans t32(clk, enable, reset, x_mid[3][1], y_mid[3][2], v_mid[2][2], clear_mid[3][1], clear_out[2], x_mid[3][2], y_mid[2][2], v_mid[3][2], clear_mid[3][2], clear_out[3], x_delay_mid[3][2], x_delay_mid[3][1]);
trans t42(clk, enable, reset, x_mid[4][1], y_mid[4][2], v_mid[3][2], clear_mid[4][1], clear_out[2], x_mid[4][2], y_mid[3][2], v_mid[4][2], clear_mid[4][2], clear_out[4], x_delay_mid[4][2], x_delay_mid[4][1]);
trans t52(clk, enable, reset, x_mid[5][1], y_mid[5][2], v_mid[4][2], clear_mid[5][1], clear_out[2], x_mid[5][2], y_mid[4][2], v_mid[5][2], clear_mid[5][2], clear_out[5], x_delay_mid[5][2], x_delay_mid[5][1]);
trans t62(clk, enable, reset, x_mid[6][1], y_mid[6][2], v_mid[5][2], clear_mid[6][1], clear_out[2], x_mid[6][2], y_mid[5][2], v_mid[6][2], clear_mid[6][2], clear_out[6], x_delay_mid[6][2], x_delay_mid[6][1]);
trans t72(clk, enable, reset, x_mid[7][1],     32'b0, v_mid[6][2],   clear_mid[7][1], clear_out[2], x_mid[7][2], y_mid[6][2],    v_out[2], clear_mid[7][2], clear_out[7], x_delay_mid[7][2], x_delay_mid[7][1]);

trans t03(clk, enable, reset, x_mid[0][2], y_mid[0][3],     v_in[3], clear_mid[0][2], clear_out[3], x_mid[0][3],    y_out[3], v_mid[0][3], clear_mid[0][3], clear_out[0], x_delay_mid[0][3], x_delay_mid[0][2]);
trans t13(clk, enable, reset, x_mid[1][2], y_mid[1][3], v_mid[0][3], clear_mid[1][2], clear_out[3], x_mid[1][3], y_mid[0][3], v_mid[1][3], clear_mid[1][3], clear_out[1], x_delay_mid[1][3], x_delay_mid[1][2]);
trans t23(clk, enable, reset, x_mid[2][2], y_mid[2][3], v_mid[1][3], clear_mid[2][2], clear_out[3], x_mid[2][3], y_mid[1][3], v_mid[2][3], clear_mid[2][3], clear_out[2], x_delay_mid[2][3], x_delay_mid[2][2]);
trans t33(clk, enable, reset, x_mid[3][2], y_mid[3][3], v_mid[2][3], clear_mid[3][2], clear_out[3], x_mid[3][3], y_mid[2][3], v_mid[3][3], clear_mid[3][3], clear_out[3], x_delay_mid[3][3], x_delay_mid[3][2]);
trans t43(clk, enable, reset, x_mid[4][2], y_mid[4][3], v_mid[3][3], clear_mid[4][2], clear_out[3], x_mid[4][3], y_mid[3][3], v_mid[4][3], clear_mid[4][3], clear_out[4], x_delay_mid[4][3], x_delay_mid[4][2]);
trans t53(clk, enable, reset, x_mid[5][2], y_mid[5][3], v_mid[4][3], clear_mid[5][2], clear_out[3], x_mid[5][3], y_mid[4][3], v_mid[5][3], clear_mid[5][3], clear_out[5], x_delay_mid[5][3], x_delay_mid[5][2]);
trans t63(clk, enable, reset, x_mid[6][2], y_mid[6][3], v_mid[5][3], clear_mid[6][2], clear_out[3], x_mid[6][3], y_mid[5][3], v_mid[6][3], clear_mid[6][3], clear_out[6], x_delay_mid[6][3], x_delay_mid[6][2]);
trans t73(clk, enable, reset, x_mid[7][2],     32'b0, v_mid[6][3],   clear_mid[7][2], clear_out[3], x_mid[7][3], y_mid[6][3],    v_out[3], clear_mid[7][3], clear_out[7], x_delay_mid[7][3], x_delay_mid[7][2]);

trans t04(clk, enable, reset, x_mid[0][3], y_mid[0][4],     v_in[4], clear_mid[0][3], clear_out[4], x_mid[0][4],    y_out[4], v_mid[0][4], clear_mid[0][4], clear_out[0], x_delay_mid[0][4], x_delay_mid[0][3]);
trans t14(clk, enable, reset, x_mid[1][3], y_mid[1][4], v_mid[0][4], clear_mid[1][3], clear_out[4], x_mid[1][4], y_mid[0][4], v_mid[1][4], clear_mid[1][4], clear_out[1], x_delay_mid[1][4], x_delay_mid[1][3]);
trans t24(clk, enable, reset, x_mid[2][3], y_mid[2][4], v_mid[1][4], clear_mid[2][3], clear_out[4], x_mid[2][4], y_mid[1][4], v_mid[2][4], clear_mid[2][4], clear_out[2], x_delay_mid[2][4], x_delay_mid[2][3]);
trans t34(clk, enable, reset, x_mid[3][3], y_mid[3][4], v_mid[2][4], clear_mid[3][3], clear_out[4], x_mid[3][4], y_mid[2][4], v_mid[3][4], clear_mid[3][4], clear_out[3], x_delay_mid[3][4], x_delay_mid[3][3]);
trans t44(clk, enable, reset, x_mid[4][3], y_mid[4][4], v_mid[3][4], clear_mid[4][3], clear_out[4], x_mid[4][4], y_mid[3][4], v_mid[4][4], clear_mid[4][4], clear_out[4], x_delay_mid[4][4], x_delay_mid[4][3]);
trans t54(clk, enable, reset, x_mid[5][3], y_mid[5][4], v_mid[4][4], clear_mid[5][3], clear_out[4], x_mid[5][4], y_mid[4][4], v_mid[5][4], clear_mid[5][4], clear_out[5], x_delay_mid[5][4], x_delay_mid[5][3]);
trans t64(clk, enable, reset, x_mid[6][3], y_mid[6][4], v_mid[5][4], clear_mid[6][3], clear_out[4], x_mid[6][4], y_mid[5][4], v_mid[6][4], clear_mid[6][4], clear_out[6], x_delay_mid[6][4], x_delay_mid[6][3]);
trans t74(clk, enable, reset, x_mid[7][3],     32'b0, v_mid[6][4],   clear_mid[7][3], clear_out[4], x_mid[7][4], y_mid[6][4],    v_out[4], clear_mid[7][4], clear_out[7], x_delay_mid[7][4], x_delay_mid[7][3]);

trans t05(clk, enable, reset, x_mid[0][4], y_mid[0][5],     v_in[5], clear_mid[0][4], clear_out[5], x_mid[0][5],    y_out[5], v_mid[0][5], clear_mid[0][5], clear_out[0], x_delay_mid[0][5], x_delay_mid[0][4]);
trans t15(clk, enable, reset, x_mid[1][4], y_mid[1][5], v_mid[0][5], clear_mid[1][4], clear_out[5], x_mid[1][5], y_mid[0][5], v_mid[1][5], clear_mid[1][5], clear_out[1], x_delay_mid[1][5], x_delay_mid[1][4]);
trans t25(clk, enable, reset, x_mid[2][4], y_mid[2][5], v_mid[1][5], clear_mid[2][4], clear_out[5], x_mid[2][5], y_mid[1][5], v_mid[2][5], clear_mid[2][5], clear_out[2], x_delay_mid[2][5], x_delay_mid[2][4]);
trans t35(clk, enable, reset, x_mid[3][4], y_mid[3][5], v_mid[2][5], clear_mid[3][4], clear_out[5], x_mid[3][5], y_mid[2][5], v_mid[3][5], clear_mid[3][5], clear_out[3], x_delay_mid[3][5], x_delay_mid[3][4]);
trans t45(clk, enable, reset, x_mid[4][4], y_mid[4][5], v_mid[3][5], clear_mid[4][4], clear_out[5], x_mid[4][5], y_mid[3][5], v_mid[4][5], clear_mid[4][5], clear_out[4], x_delay_mid[4][5], x_delay_mid[4][4]);
trans t55(clk, enable, reset, x_mid[5][4], y_mid[5][5], v_mid[4][5], clear_mid[5][4], clear_out[5], x_mid[5][5], y_mid[4][5], v_mid[5][5], clear_mid[5][5], clear_out[5], x_delay_mid[5][5], x_delay_mid[5][4]);
trans t65(clk, enable, reset, x_mid[6][4], y_mid[6][5], v_mid[5][5], clear_mid[6][4], clear_out[5], x_mid[6][5], y_mid[5][5], v_mid[6][5], clear_mid[6][5], clear_out[6], x_delay_mid[6][5], x_delay_mid[6][4]);
trans t75(clk, enable, reset, x_mid[7][4],     32'b0, v_mid[6][5],   clear_mid[7][4], clear_out[5], x_mid[7][5], y_mid[6][5],    v_out[5], clear_mid[7][5], clear_out[7], x_delay_mid[7][5], x_delay_mid[7][4]);

trans t06(clk, enable, reset, x_mid[0][5], y_mid[0][6],     v_in[6], clear_mid[0][5], clear_out[6], x_mid[0][6],    y_out[6], v_mid[0][6], clear_mid[0][6], clear_out[0], x_delay_mid[0][6], x_delay_mid[0][5]);
trans t16(clk, enable, reset, x_mid[1][5], y_mid[1][6], v_mid[0][6], clear_mid[1][5], clear_out[6], x_mid[1][6], y_mid[0][6], v_mid[1][6], clear_mid[1][6], clear_out[1], x_delay_mid[1][6], x_delay_mid[1][5]);
trans t26(clk, enable, reset, x_mid[2][5], y_mid[2][6], v_mid[1][6], clear_mid[2][5], clear_out[6], x_mid[2][6], y_mid[1][6], v_mid[2][6], clear_mid[2][6], clear_out[2], x_delay_mid[2][6], x_delay_mid[2][5]);
trans t36(clk, enable, reset, x_mid[3][5], y_mid[3][6], v_mid[2][6], clear_mid[3][5], clear_out[6], x_mid[3][6], y_mid[2][6], v_mid[3][6], clear_mid[3][6], clear_out[3], x_delay_mid[3][6], x_delay_mid[3][5]);
trans t46(clk, enable, reset, x_mid[4][5], y_mid[4][6], v_mid[3][6], clear_mid[4][5], clear_out[6], x_mid[4][6], y_mid[3][6], v_mid[4][6], clear_mid[4][6], clear_out[4], x_delay_mid[4][6], x_delay_mid[4][5]);
trans t56(clk, enable, reset, x_mid[5][5], y_mid[5][6], v_mid[4][6], clear_mid[5][5], clear_out[6], x_mid[5][6], y_mid[4][6], v_mid[5][6], clear_mid[5][6], clear_out[5], x_delay_mid[5][6], x_delay_mid[5][5]);
trans t66(clk, enable, reset, x_mid[6][5], y_mid[6][6], v_mid[5][6], clear_mid[6][5], clear_out[6], x_mid[6][6], y_mid[5][6], v_mid[6][6], clear_mid[6][6], clear_out[6], x_delay_mid[6][6], x_delay_mid[6][5]);
trans t76(clk, enable, reset, x_mid[7][5],     32'b0, v_mid[6][6],   clear_mid[7][5], clear_out[6], x_mid[7][6], y_mid[6][6],    v_out[6], clear_mid[7][6], clear_out[7], x_delay_mid[7][6], x_delay_mid[7][5]);

trans t07(clk, enable, reset, x_mid[0][6], y_mid[0][7],     v_in[7], clear_mid[0][6], clear_out[7], x_out[0],    y_out[7], v_mid[0][7], clear_out[0], clear_out[0], 33'b0, x_delay_mid[0][6]);
trans t17(clk, enable, reset, x_mid[1][6], y_mid[1][7], v_mid[0][7], clear_mid[1][6], clear_out[7], x_out[1], y_mid[0][7], v_mid[1][7], clear_out[1], clear_out[1], 33'b0, x_delay_mid[1][6]);
trans t27(clk, enable, reset, x_mid[2][6], y_mid[2][7], v_mid[1][7], clear_mid[2][6], clear_out[7], x_out[2], y_mid[1][7], v_mid[2][7], clear_out[2], clear_out[2], 33'b0, x_delay_mid[2][6]);
trans t37(clk, enable, reset, x_mid[3][6], y_mid[3][7], v_mid[2][7], clear_mid[3][6], clear_out[7], x_out[3], y_mid[2][7], v_mid[3][7], clear_out[3], clear_out[3], 33'b0, x_delay_mid[3][6]);
trans t47(clk, enable, reset, x_mid[4][6], y_mid[4][7], v_mid[3][7], clear_mid[4][6], clear_out[7], x_out[4], y_mid[3][7], v_mid[4][7], clear_out[4], clear_out[4], 33'b0, x_delay_mid[4][6]);
trans t57(clk, enable, reset, x_mid[5][6], y_mid[5][7], v_mid[4][7], clear_mid[5][6], clear_out[7], x_out[5], y_mid[4][7], v_mid[5][7], clear_out[5], clear_out[5], 33'b0, x_delay_mid[5][6]);
trans t67(clk, enable, reset, x_mid[6][6], y_mid[6][7], v_mid[5][7], clear_mid[6][6], clear_out[7], x_out[6], y_mid[5][7], v_mid[6][7], clear_out[6], clear_out[6], 33'b0, x_delay_mid[6][6]);
trans t77(clk, enable, reset, x_mid[7][6],     32'b0, v_mid[6][7],   clear_mid[7][6], clear_out[7], x_out[7], y_mid[6][7],    v_out[7], clear_out[7], clear_out[7], 33'b0, x_delay_mid[7][6]);

endmodule
