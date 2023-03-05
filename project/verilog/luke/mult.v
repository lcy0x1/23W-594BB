`include "calc.v"

/*
Author: Arthur Wang
Creation Date: Nov 13 
Last Modified: Nov 29

module mac: multiply and accumulate module
module m8x8: 8 by 8 matrix multiplication module

->  clk: clock
->  enable: global enable, nothing shall be done if it is low
->  reset: reset everything
->  config: configurations

->  win_raw: vertical input
->  xin_raw: horizontal input
->  clear_in_raw: clear signal that clears accumulator
<-  z_out: output of shift register
<-  clear_out: place to shift out clear signal, typically used for <shift>
<-  b_out: binary relu derivative output

How it works:
                 w_in
                   |
--------------+----|----------- shift
              |    v
              +-> [   ]
x_in, clear_in -> [mac] -> x_out, clear_out
         y_out <- [   ] <- y_in
                   |
                   v
                w_out

config[1]: relu, pass data through relu
config[2]: transpose w
config[3]: transpose x

*/

module mac(
  input [31:0] w_in,
  input [31:0] x_in,
  input [31:0] y_in,
  input clear_in,
  input enable,
  input shift,
  input clk,
  input reset,
  output reg [31:0] w_out,
  output reg [31:0] x_out,
  output reg [31:0] y_out,
  output reg clear_out
);
  
  reg [31:0] acc;
  reg [31:0] standby;
  
  wire [31:0] wx;
  wire [31:0] sum;
  wire [31:0] preg = clear_in ? 0 : acc;
  multiplier m0(w_in, x_in, wx);
  adder m1(wx, preg, sum);
  
  always @(posedge clk) begin
    if(enable || reset) begin
      w_out <= reset ? 0 : w_in;
      x_out <= reset ? 0 : x_in;
      clear_out <= reset ? 0 : clear_in;
      acc <= reset ? 0 : sum;
      standby <= reset ? 0 : clear_in ? acc : standby;
      y_out <= reset ? 0 : shift ? standby : y_in;
    end
  end
  
endmodule

module m8x8(
  input clk,
  input reset,
  input enable,
  input en,
  input [3:0] conf,
  input [31:0] win_raw [7:0],
  input [31:0] xin_raw [7:0],
  input [7:0] clear_in_raw,
  output [31:0] z_out [7:0],
  output [7:0] clear_out,
  output [7:0] b_out
);

  wire relu = conf[1];
  wire wt = conf[2];
  wire xt = conf[3];

  wire [7:0] temp_0;


  wire [31:0] w_in [7:0];
  wire [31:0] x_in [7:0];
  wire [7:0] clear_in;

  t8x8 transw(clk, enable, reset, wt,  win_raw, en, w_in, clear_in_raw, clear_in);
  t8x8 transx(clk, enable, reset, xt,  xin_raw, en, x_in, clear_in_raw, temp_0);
  
  wire [31:0] w_mid [6:0][7:0];
  wire [31:0] x_mid [7:0][6:0];
  wire [31:0] y_mid [7:0][6:0];
  wire clear_mid [7:0][6:0];


  wire [31:0] y_out[7:0];
  wire [31:0] a_out[7:0];

  wire [31:0] w_out [7:0];
  wire [31:0] x_out [7:0];

  relu r0(y_out[0], a_out[0], b_out[0]);
  relu r1(y_out[1], a_out[1], b_out[1]);
  relu r2(y_out[2], a_out[2], b_out[2]);
  relu r3(y_out[3], a_out[3], b_out[3]);
  relu r4(y_out[4], a_out[4], b_out[4]);
  relu r5(y_out[5], a_out[5], b_out[5]);
  relu r6(y_out[6], a_out[6], b_out[6]);
  relu r7(y_out[7], a_out[7], b_out[7]);

  assign z_out[0] = relu ? a_out[0] : y_out[0];
  assign z_out[1] = relu ? a_out[1] : y_out[1];
  assign z_out[2] = relu ? a_out[2] : y_out[2];
  assign z_out[3] = relu ? a_out[3] : y_out[3];
  assign z_out[4] = relu ? a_out[4] : y_out[4];
  assign z_out[5] = relu ? a_out[5] : y_out[5];
  assign z_out[6] = relu ? a_out[6] : y_out[6];
  assign z_out[7] = relu ? a_out[7] : y_out[7];

  // up: w_in
  // down: w_out
  // left: x_in, y_out, clear_in
  // right: x_out, y_in, clear_out, shift
  
  mac m00(w_in[0]    , x_in[0], y_mid[0][0], clear_in[0], enable, clear_out[0], clk, reset, w_mid[0][0], x_mid[0][0], y_out[0], clear_mid[0][0]);
  mac m10(w_mid[0][0], x_in[1], y_mid[1][0], clear_in[1], enable, clear_out[1], clk, reset, w_mid[1][0], x_mid[1][0], y_out[1], clear_mid[1][0]);
  mac m20(w_mid[1][0], x_in[2], y_mid[2][0], clear_in[2], enable, clear_out[2], clk, reset, w_mid[2][0], x_mid[2][0], y_out[2], clear_mid[2][0]);
  mac m30(w_mid[2][0], x_in[3], y_mid[3][0], clear_in[3], enable, clear_out[3], clk, reset, w_mid[3][0], x_mid[3][0], y_out[3], clear_mid[3][0]);
  mac m40(w_mid[3][0], x_in[4], y_mid[4][0], clear_in[4], enable, clear_out[4], clk, reset, w_mid[4][0], x_mid[4][0], y_out[4], clear_mid[4][0]);
  mac m50(w_mid[4][0], x_in[5], y_mid[5][0], clear_in[5], enable, clear_out[5], clk, reset, w_mid[5][0], x_mid[5][0], y_out[5], clear_mid[5][0]);
  mac m60(w_mid[5][0], x_in[6], y_mid[6][0], clear_in[6], enable, clear_out[6], clk, reset, w_mid[6][0], x_mid[6][0], y_out[6], clear_mid[6][0]);
  mac m70(w_mid[6][0], x_in[7], y_mid[7][0], clear_in[7], enable, clear_out[7], clk, reset, w_out[0],    x_mid[7][0], y_out[7], clear_mid[7][0]);
  
  mac m01(w_in[1]    , x_mid[0][0], y_mid[0][1], clear_mid[0][0], enable, clear_out[0], clk, reset, w_mid[0][1], x_mid[0][1], y_mid[0][0], clear_mid[0][1]);
  mac m11(w_mid[0][1], x_mid[1][0], y_mid[1][1], clear_mid[1][0], enable, clear_out[1], clk, reset, w_mid[1][1], x_mid[1][1], y_mid[1][0], clear_mid[1][1]);
  mac m21(w_mid[1][1], x_mid[2][0], y_mid[2][1], clear_mid[2][0], enable, clear_out[2], clk, reset, w_mid[2][1], x_mid[2][1], y_mid[2][0], clear_mid[2][1]);
  mac m31(w_mid[2][1], x_mid[3][0], y_mid[3][1], clear_mid[3][0], enable, clear_out[3], clk, reset, w_mid[3][1], x_mid[3][1], y_mid[3][0], clear_mid[3][1]);
  mac m41(w_mid[3][1], x_mid[4][0], y_mid[4][1], clear_mid[4][0], enable, clear_out[4], clk, reset, w_mid[4][1], x_mid[4][1], y_mid[4][0], clear_mid[4][1]);
  mac m51(w_mid[4][1], x_mid[5][0], y_mid[5][1], clear_mid[5][0], enable, clear_out[5], clk, reset, w_mid[5][1], x_mid[5][1], y_mid[5][0], clear_mid[5][1]);
  mac m61(w_mid[5][1], x_mid[6][0], y_mid[6][1], clear_mid[6][0], enable, clear_out[6], clk, reset, w_mid[6][1], x_mid[6][1], y_mid[6][0], clear_mid[6][1]);
  mac m71(w_mid[6][1], x_mid[7][0], y_mid[7][1], clear_mid[7][0], enable, clear_out[7], clk, reset,    w_out[1], x_mid[7][1], y_mid[7][0], clear_mid[7][1]);
  
  mac m02(w_in[2]    , x_mid[0][1], y_mid[0][2], clear_mid[0][1], enable, clear_out[0], clk, reset, w_mid[0][2], x_mid[0][2], y_mid[0][1], clear_mid[0][2]);
  mac m12(w_mid[0][2], x_mid[1][1], y_mid[1][2], clear_mid[1][1], enable, clear_out[1], clk, reset, w_mid[1][2], x_mid[1][2], y_mid[1][1], clear_mid[1][2]);
  mac m22(w_mid[1][2], x_mid[2][1], y_mid[2][2], clear_mid[2][1], enable, clear_out[2], clk, reset, w_mid[2][2], x_mid[2][2], y_mid[2][1], clear_mid[2][2]);
  mac m32(w_mid[2][2], x_mid[3][1], y_mid[3][2], clear_mid[3][1], enable, clear_out[3], clk, reset, w_mid[3][2], x_mid[3][2], y_mid[3][1], clear_mid[3][2]);
  mac m42(w_mid[3][2], x_mid[4][1], y_mid[4][2], clear_mid[4][1], enable, clear_out[4], clk, reset, w_mid[4][2], x_mid[4][2], y_mid[4][1], clear_mid[4][2]);
  mac m52(w_mid[4][2], x_mid[5][1], y_mid[5][2], clear_mid[5][1], enable, clear_out[5], clk, reset, w_mid[5][2], x_mid[5][2], y_mid[5][1], clear_mid[5][2]);
  mac m62(w_mid[5][2], x_mid[6][1], y_mid[6][2], clear_mid[6][1], enable, clear_out[6], clk, reset, w_mid[6][2], x_mid[6][2], y_mid[6][1], clear_mid[6][2]);
  mac m72(w_mid[6][2], x_mid[7][1], y_mid[7][2], clear_mid[7][1], enable, clear_out[7], clk, reset,    w_out[2], x_mid[7][2], y_mid[7][1], clear_mid[7][2]);
  
  mac m03(w_in[3]    , x_mid[0][2], y_mid[0][3], clear_mid[0][2], enable, clear_out[0], clk, reset, w_mid[0][3], x_mid[0][3], y_mid[0][2], clear_mid[0][3]);
  mac m13(w_mid[0][3], x_mid[1][2], y_mid[1][3], clear_mid[1][2], enable, clear_out[1], clk, reset, w_mid[1][3], x_mid[1][3], y_mid[1][2], clear_mid[1][3]);
  mac m23(w_mid[1][3], x_mid[2][2], y_mid[2][3], clear_mid[2][2], enable, clear_out[2], clk, reset, w_mid[2][3], x_mid[2][3], y_mid[2][2], clear_mid[2][3]);
  mac m33(w_mid[2][3], x_mid[3][2], y_mid[3][3], clear_mid[3][2], enable, clear_out[3], clk, reset, w_mid[3][3], x_mid[3][3], y_mid[3][2], clear_mid[3][3]);
  mac m43(w_mid[3][3], x_mid[4][2], y_mid[4][3], clear_mid[4][2], enable, clear_out[4], clk, reset, w_mid[4][3], x_mid[4][3], y_mid[4][2], clear_mid[4][3]);
  mac m53(w_mid[4][3], x_mid[5][2], y_mid[5][3], clear_mid[5][2], enable, clear_out[5], clk, reset, w_mid[5][3], x_mid[5][3], y_mid[5][2], clear_mid[5][3]);
  mac m63(w_mid[5][3], x_mid[6][2], y_mid[6][3], clear_mid[6][2], enable, clear_out[6], clk, reset, w_mid[6][3], x_mid[6][3], y_mid[6][2], clear_mid[6][3]);
  mac m73(w_mid[6][3], x_mid[7][2], y_mid[7][3], clear_mid[7][2], enable, clear_out[7], clk, reset,    w_out[3], x_mid[7][3], y_mid[7][2], clear_mid[7][3]);

  mac m04(w_in[4]    , x_mid[0][3], y_mid[0][4], clear_mid[0][3], enable, clear_out[0], clk, reset, w_mid[0][4], x_mid[0][4], y_mid[0][3], clear_mid[0][4]);
  mac m14(w_mid[0][4], x_mid[1][3], y_mid[1][4], clear_mid[1][3], enable, clear_out[1], clk, reset, w_mid[1][4], x_mid[1][4], y_mid[1][3], clear_mid[1][4]);
  mac m24(w_mid[1][4], x_mid[2][3], y_mid[2][4], clear_mid[2][3], enable, clear_out[2], clk, reset, w_mid[2][4], x_mid[2][4], y_mid[2][3], clear_mid[2][4]);
  mac m34(w_mid[2][4], x_mid[3][3], y_mid[3][4], clear_mid[3][3], enable, clear_out[3], clk, reset, w_mid[3][4], x_mid[3][4], y_mid[3][3], clear_mid[3][4]);
  mac m44(w_mid[3][4], x_mid[4][3], y_mid[4][4], clear_mid[4][3], enable, clear_out[4], clk, reset, w_mid[4][4], x_mid[4][4], y_mid[4][3], clear_mid[4][4]);
  mac m54(w_mid[4][4], x_mid[5][3], y_mid[5][4], clear_mid[5][3], enable, clear_out[5], clk, reset, w_mid[5][4], x_mid[5][4], y_mid[5][3], clear_mid[5][4]);
  mac m64(w_mid[5][4], x_mid[6][3], y_mid[6][4], clear_mid[6][3], enable, clear_out[6], clk, reset, w_mid[6][4], x_mid[6][4], y_mid[6][3], clear_mid[6][4]);
  mac m74(w_mid[6][4], x_mid[7][3], y_mid[7][4], clear_mid[7][3], enable, clear_out[7], clk, reset,    w_out[4], x_mid[7][4], y_mid[7][3], clear_mid[7][4]);

  mac m05(w_in[5]    , x_mid[0][4], y_mid[0][5], clear_mid[0][4], enable, clear_out[0], clk, reset, w_mid[0][5], x_mid[0][5], y_mid[0][4], clear_mid[0][5]);
  mac m15(w_mid[0][5], x_mid[1][4], y_mid[1][5], clear_mid[1][4], enable, clear_out[1], clk, reset, w_mid[1][5], x_mid[1][5], y_mid[1][4], clear_mid[1][5]);
  mac m25(w_mid[1][5], x_mid[2][4], y_mid[2][5], clear_mid[2][4], enable, clear_out[2], clk, reset, w_mid[2][5], x_mid[2][5], y_mid[2][4], clear_mid[2][5]);
  mac m35(w_mid[2][5], x_mid[3][4], y_mid[3][5], clear_mid[3][4], enable, clear_out[3], clk, reset, w_mid[3][5], x_mid[3][5], y_mid[3][4], clear_mid[3][5]);
  mac m45(w_mid[3][5], x_mid[4][4], y_mid[4][5], clear_mid[4][4], enable, clear_out[4], clk, reset, w_mid[4][5], x_mid[4][5], y_mid[4][4], clear_mid[4][5]);
  mac m55(w_mid[4][5], x_mid[5][4], y_mid[5][5], clear_mid[5][4], enable, clear_out[5], clk, reset, w_mid[5][5], x_mid[5][5], y_mid[5][4], clear_mid[5][5]);
  mac m65(w_mid[5][5], x_mid[6][4], y_mid[6][5], clear_mid[6][4], enable, clear_out[6], clk, reset, w_mid[6][5], x_mid[6][5], y_mid[6][4], clear_mid[6][5]);
  mac m75(w_mid[6][5], x_mid[7][4], y_mid[7][5], clear_mid[7][4], enable, clear_out[7], clk, reset,    w_out[5], x_mid[7][5], y_mid[7][4], clear_mid[7][5]);
    
  mac m06(w_in[6]    , x_mid[0][5], y_mid[0][6], clear_mid[0][5], enable, clear_out[0], clk, reset, w_mid[0][6], x_mid[0][6], y_mid[0][5], clear_mid[0][6]);
  mac m16(w_mid[0][6], x_mid[1][5], y_mid[1][6], clear_mid[1][5], enable, clear_out[1], clk, reset, w_mid[1][6], x_mid[1][6], y_mid[1][5], clear_mid[1][6]);
  mac m26(w_mid[1][6], x_mid[2][5], y_mid[2][6], clear_mid[2][5], enable, clear_out[2], clk, reset, w_mid[2][6], x_mid[2][6], y_mid[2][5], clear_mid[2][6]);
  mac m36(w_mid[2][6], x_mid[3][5], y_mid[3][6], clear_mid[3][5], enable, clear_out[3], clk, reset, w_mid[3][6], x_mid[3][6], y_mid[3][5], clear_mid[3][6]);
  mac m46(w_mid[3][6], x_mid[4][5], y_mid[4][6], clear_mid[4][5], enable, clear_out[4], clk, reset, w_mid[4][6], x_mid[4][6], y_mid[4][5], clear_mid[4][6]);
  mac m56(w_mid[4][6], x_mid[5][5], y_mid[5][6], clear_mid[5][5], enable, clear_out[5], clk, reset, w_mid[5][6], x_mid[5][6], y_mid[5][5], clear_mid[5][6]);
  mac m66(w_mid[5][6], x_mid[6][5], y_mid[6][6], clear_mid[6][5], enable, clear_out[6], clk, reset, w_mid[6][6], x_mid[6][6], y_mid[6][5], clear_mid[6][6]);
  mac m76(w_mid[6][6], x_mid[7][5], y_mid[7][6], clear_mid[7][5], enable, clear_out[7], clk, reset,    w_out[6], x_mid[7][6], y_mid[7][5], clear_mid[7][6]);
  
  mac m07(w_in[7]    , x_mid[0][6], 32'b0, clear_mid[0][6], enable, clear_out[0], clk, reset, w_mid[0][7], x_out[0], y_mid[0][6], clear_out[0]);
  mac m17(w_mid[0][7], x_mid[1][6], 32'b0, clear_mid[1][6], enable, clear_out[1], clk, reset, w_mid[1][7], x_out[1], y_mid[1][6], clear_out[1]);
  mac m27(w_mid[1][7], x_mid[2][6], 32'b0, clear_mid[2][6], enable, clear_out[2], clk, reset, w_mid[2][7], x_out[2], y_mid[2][6], clear_out[2]);
  mac m37(w_mid[2][7], x_mid[3][6], 32'b0, clear_mid[3][6], enable, clear_out[3], clk, reset, w_mid[3][7], x_out[3], y_mid[3][6], clear_out[3]);
  mac m47(w_mid[3][7], x_mid[4][6], 32'b0, clear_mid[4][6], enable, clear_out[4], clk, reset, w_mid[4][7], x_out[4], y_mid[4][6], clear_out[4]);
  mac m57(w_mid[4][7], x_mid[5][6], 32'b0, clear_mid[5][6], enable, clear_out[5], clk, reset, w_mid[5][7], x_out[5], y_mid[5][6], clear_out[5]);
  mac m67(w_mid[5][7], x_mid[6][6], 32'b0, clear_mid[6][6], enable, clear_out[6], clk, reset, w_mid[6][7], x_out[6], y_mid[6][6], clear_out[6]);
  mac m77(w_mid[6][7], x_mid[7][6], 32'b0, clear_mid[7][6], enable, clear_out[7], clk, reset,    w_out[7], x_out[7], y_mid[7][6], clear_out[7]);
  
endmodule