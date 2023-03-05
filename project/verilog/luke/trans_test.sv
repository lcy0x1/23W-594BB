// Code your testbench here
// or browse Examples
module testbench();
  
  parameter CLK = 4;
  
  reg enable;
  reg clk;
  reg reset;

  reg [31:0] x_in [7:0];
  reg [31:0] y_in [7:0]; 
  reg [7:0] v_in;
  reg [7:0] clear_in;
  reg [7:0] shift;
  wire [31:0] x_out [7:0];
  wire [31:0] y_out [7:0];
  wire [7:0] v_out;
  wire [7:0] clear_out;

  t8x8 main(clk, enable, reset, x_in, y_in, v_in, clear_in, shift, x_out, y_out, v_out, clear_out);
//   reg [31:0] x_in;
//   reg [31:0] y_in;
//   reg v_in;
//   reg clear_in;
//   reg shift;
//   wire [31:0] x_out;
//   wire [31:0] y_out;
//   wire v_out;
//   wire clear_out;

//   trans main(clk, enable, reset, x_in, y_in, v_in, clear_in, shift, x_out, y_out, v_out, clear_out);
//   initial begin
//     #1;
//     forever begin
//       clk = ~clk;
//       #2;
//     end
//   end
  
  initial begin
    $dumpfile ("dump.vcd");
    $dumpvars;
    clk = 0;
    reset = 0;
    enable = 0; 
    // x_in = 0;
    // y_in = 0;
    // v_in = 0;
    // #(CLK);
    // reset = 1;
    // #(CLK);
    // reset = 0;
    // enable = 1;
    // #(CLK);
    
    $finish;
  end
  
endmodule
