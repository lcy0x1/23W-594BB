`include "control.v"
/*
Author: Arthur Wang
Create Date: Nov 19
Edit Date: Dec 5

Outer Interface of verilog modules

->  clk: global clock
->  clear: global reset
->  enable: global enable for verilog modules
->  data_in: the only input bus
<-  data_out: the only output bus
<-  y_valid: valid flag for output data
<-  out_count: for every set of inputs, tells the expected number of output
<- out_count_valid: valid flag for out_count

Expected data format for data_interface:
number of operations
expected output length
data

*/

module data_interface(
    input clk,
    input clear,
    input [31:0] data_in,
    input enable,
    output reg [31:0] data_out,
    output reg y_valid,
    output reg [31:0] out_count,
    output reg out_count_valid,
    output reg ready
);

  reg [31:0] counter;
  reg [31:0] temp_op;
  reg [31:0] operation;
  reg [31:0] data;
  reg [31:0] out_count_temp;
  reg [8:0] size;

  wire [31:0] out_data;

  controller main(clk, enable, clear, operation, data, out_data, size);

  always @(posedge clk) begin
    if(clear) begin
      counter <= 0;
      temp_op <= 0;
      operation <= 0;
      data <= 0;
      data_out <= 0;
      y_valid <= 0;
      out_count <= 0;
      out_count_valid <= 0;
      ready <= 1;
      size <= 9'b001001111; // 16x16
    end else if(enable) begin
      if(counter == 0) begin
        if(data_in[19:16] == 0) begin
          counter <= data_in[15:0];
        end else if(data_in[19:16] == 1 && !out_count_valid) begin
          out_count <= data_in[15:0];
          out_count_valid <= 1;
        end else if(data_in[19:16] == 2) begin
          size <= data_in[8:0];
        end
        temp_op <= 0;
        operation <= 0;
        data <= 0;
        ready <= 1;
      end else if(temp_op == 0) begin
        temp_op <= data_in;
        data <= 0;
        ready <= data_in[3:0] == 2;
      end else begin
        operation <= temp_op;
        data <= data_in;
        counter <= counter - 1;    
        ready <= data_in[3:0] == 2;
      end
      data_out <= out_data;
      y_valid <= operation[3:0] == 3;
      if(out_count_valid) begin
        out_count <= 0;
        out_count_valid <= 0;
      end
    end
  end

endmodule