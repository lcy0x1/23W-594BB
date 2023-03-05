/*
Author: Arthur Wang
Creation Date: Nov 13 
Last Modified: Dec 5

This files contains interface of basic operations,
allowing to be replaced by floating point operations

*/

module multiplier(
  input signed [31:0] a,
  input signed [31:0] b,
  output signed [31:0] y
);
  wire [63:0] z = a*b;
  assign y = z[47:16] + z[15];
endmodule

module adder(
  input signed [31:0] a,
  input signed [31:0] b,
  output signed [31:0] y
);
  assign y = a+b;
endmodule

module relu(
  input signed [31:0] a,
  output signed [31:0] y,
  output d
);
  assign y = a > 0 ? a : 0;
  assign d = a > 0;
endmodule