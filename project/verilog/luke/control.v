`include "mult.v"
`include "blockmem.v"
`include "transpose.v"

/*
Author: Arthur Wang, Ian Wu
Creation Date: Nov 14 
Last Modified: Dec 5

TODO: use hadamard product
TODO: additive update as configuration for op-code = 1
TODO: allow non-square and non 2^N size matrix multiplication

->  clk: clock
->  enable: global enable, nothing shall be done if it is low
->  reset: global reset, clears everything except memory
->  operation: operation to be done
->  in_data: input data, for reading matrix only

Page Addresses:
4 bits, highest 2 bits tells which register file to use.
Lower 2 bits is for page number

Operations: typically segregated by chunk of 4 bits
chunk[0]: op code
op code == 0: idle
op code == 1: calculate A*B'
  chunk[1] = A page number (to read)
  chunk[2] = B page number (to read)
  chunk[3] = result page number (to write in bulk)
  chunk[4] = configuration {transpose A, transpose B, use ReLU, unused}
  chunk[5] = relu derivative target
  chunk[6] = configuration {unused, unused, add data to memory, save hadamard product on write}
op code == 2: calculate A⊙B'
  chunk[1] = destination page number (to write in serial)
  chunk[2] = configuration {unused, unused, add data to memory, save hadamard product on write}
  chunk[3] = 
op code == 3:
  chunk[1] = source page number (to read in serial)


*/

module controller(
  input clk,
  input enable,
  input reset,
  input [31:0] operation,
  input [31:0] in_data,
  output [31:0] out_data,
  input [8:0] size
);
  
  // decode wire
  wire [3:0] opcode = operation[3:0];
  wire [3:0] op_a = operation[7:4];
  wire [3:0] op_b = operation[11:8];
  wire [3:0] op_c = operation[15:12];
  wire [3:0] op_d = operation[19:16];
  wire [3:0] op_e = operation[23:20];
  wire [3:0] op_f = operation[27:24];
  
  // write enable for register file

// if(opcode == 2){
//   if(op_a[3:2] == 0){
//     return 1;
//   } else {
//     return 0;
//   }
// } else if(opcode == 1){
//   if(op_c[3:2] == 0){
//     if(op_f[0] == 1){
//       return 4;
//      } else if(op_f[1] == 1){
//        return 3;
//      } else {
//        return 2;
//      }
//   } else if(op_e[3:2] == 0){
//     return 2;
//   } else {
//     return 0;
//   }
// } else {
//   return 0;
// }


  wire [2:0] we_0 = opcode == 2 ? op_a[3:2] == 0 ? {1'b0, op_b[0]? 2'b10 : op_b[1]? 2'b11 : 2'b01} : 0 : opcode == 1 ? op_c[3:2] == 0 ? {1'b1, op_f[0]? 2'b10 : op_f[1]? 2'b11 : 2'b01} : op_d[1] && (op_e[3:2] == 0) ? 3'b101 : 0 : 0;
  wire [2:0] we_1 = opcode == 2 ? op_a[3:2] == 1 ? {1'b0, op_b[0]? 2'b10 : op_b[1]? 2'b11 : 2'b01} : 0 : opcode == 1 ? op_c[3:2] == 1 ? {1'b1, op_f[0]? 2'b10 : op_f[1]? 2'b11 : 2'b01} : op_d[1] && (op_e[3:2] == 1) ? 3'b101 : 0 : 0;
  wire [2:0] we_2 = opcode == 2 ? op_a[3:2] == 2 ? {1'b0, op_b[0]? 2'b10 : op_b[1]? 2'b11 : 2'b01} : 0 : opcode == 1 ? op_c[3:2] == 2 ? {1'b1, op_f[0]? 2'b10 : op_f[1]? 2'b11 : 2'b01} : op_d[1] && (op_e[3:2] == 2) ? 3'b101 : 0 : 0;
  wire [2:0] we_3 = opcode == 2 ? op_a[3:2] == 3 ? {1'b0, op_b[0]? 2'b10 : op_b[1]? 2'b11 : 2'b01} : 0 : opcode == 1 ? op_c[3:2] == 3 ? {1'b1, op_f[0]? 2'b10 : op_f[1]? 2'b11 : 2'b01} : op_d[1] && (op_e[3:2] == 3) ? 3'b101 : 0 : 0;

  // write address for register file
  wire [1:0] wp_0 = opcode == 1 ? op_c[3:2] == 0 ? op_c[1:0] : op_e[3:2] == 0 ? op_e[1:0] : 0 : opcode == 2 ? op_a[3:2] == 0 ? op_a[1:0] : 0 : 0;
  wire [1:0] wp_1 = opcode == 1 ? op_c[3:2] == 1 ? op_c[1:0] : op_e[3:2] == 1 ? op_e[1:0] : 0 : opcode == 2 ? op_a[3:2] == 1 ? op_a[1:0] : 0 : 0;
  wire [1:0] wp_2 = opcode == 1 ? op_c[3:2] == 2 ? op_c[1:0] : op_e[3:2] == 2 ? op_e[1:0] : 0 : opcode == 2 ? op_a[3:2] == 2 ? op_a[1:0] : 0 : 0;
  wire [1:0] wp_3 = opcode == 1 ? op_c[3:2] == 3 ? op_c[1:0] : op_e[3:2] == 3 ? op_e[1:0] : 0 : opcode == 2 ? op_a[3:2] == 3 ? op_a[1:0] : 0 : 0;

  // read address for register file
  wire [1:0] rp_0 = opcode == 1 ? op_a[3:2] == 0 ? op_a[1:0] : op_b[3:2] == 0 ? op_b[1:0] : 0 : opcode == 3 ? op_a[3:2] == 0 ? op_a[1:0] : 0 : 0;
  wire [1:0] rp_1 = opcode == 1 ? op_a[3:2] == 1 ? op_a[1:0] : op_b[3:2] == 1 ? op_b[1:0] : 0 : opcode == 3 ? op_a[3:2] == 1 ? op_a[1:0] : 0 : 0;
  wire [1:0] rp_2 = opcode == 1 ? op_a[3:2] == 2 ? op_a[1:0] : op_b[3:2] == 2 ? op_b[1:0] : 0 : opcode == 3 ? op_a[3:2] == 2 ? op_a[1:0] : 0 : 0;
  wire [1:0] rp_3 = opcode == 1 ? op_a[3:2] == 3 ? op_a[1:0] : op_b[3:2] == 3 ? op_b[1:0] : 0 : opcode == 3 ? op_a[3:2] == 3 ? op_a[1:0] : 0 : 0;
  
  
  // enable flag for starting matrix multiplier (starting shifting data into multiplier)
  // it should be on only for the duration that is required for memory to shift data
  // so it is configured in such way that it is off BEFORE <opcode == 1> goes off
  wire en;
  // delayed flag ofr <op_code == 1>, so that we can know the upper edge
  reg old_en;
  // cell index for W matrix, incremented every cycle
  reg [8:0] ind_wc;
  // line index for W matrix, incremented only after <ind_wc> completes 1 cycle
  reg [2:0] ind_wl;
  // line index for X matrix, incremented only after <ind_wl> completes 1 cycle
  reg [2:0] ind_xl;

  // read mode for register file
  wire [1:0] re_x = op_d[3] ? 3 : 1;
  wire [1:0] re_w = op_d[2] ? 3 : 1;
  wire [1:0] re_0 = opcode == 3 ? op_a[3:2] == 0 ? 2 : 0 : opcode == 1 && en ? op_a[3:2] == 0 ? re_x : op_b[3:2] == 0 ? re_w : 0 : 0;
  wire [1:0] re_1 = opcode == 3 ? op_a[3:2] == 1 ? 2 : 0 : opcode == 1 && en ? op_a[3:2] == 1 ? re_x : op_b[3:2] == 1 ? re_w : 0 : 0;
  wire [1:0] re_2 = opcode == 3 ? op_a[3:2] == 2 ? 2 : 0 : opcode == 1 && en ? op_a[3:2] == 2 ? re_x : op_b[3:2] == 2 ? re_w : 0 : 0;
  wire [1:0] re_3 = opcode == 3 ? op_a[3:2] == 3 ? 2 : 0 : opcode == 1 && en ? op_a[3:2] == 3 ? re_x : op_b[3:2] == 3 ? re_w : 0 : 0;
  
  wire [31:0] out_data_0;
  wire [31:0] out_data_1;
  wire [31:0] out_data_2;
  wire [31:0] out_data_3;
  wire [31:0] chunk_read_0 [7:0];
  wire [31:0] chunk_read_1 [7:0];
  wire [31:0] chunk_read_2 [7:0];
  wire [31:0] chunk_read_3 [7:0];
  wire [7:0] clear_in_0;
  wire [7:0] clear_in_1;
  wire [7:0] clear_in_2;
  wire [7:0] clear_in_3;
  wire [31:0] chunk_write_0 [7:0];
  wire [31:0] chunk_write_1 [7:0];
  wire [31:0] chunk_write_2 [7:0];
  wire [31:0] chunk_write_3 [7:0];  
  wire switch_0;
  wire switch_1;
  wire switch_2;
  wire switch_3;

  // wires connecting memory and multiplier
  wire [31:0] w_in [7:0];
  wire [31:0] x_in [7:0];
  wire [7:0] clear_in;
  wire [31:0] y_out [7:0];
  wire [7:0] clear_out;
  wire [7:0] b_out;

  // output valid flag is 1 cycle delay of clear_out
  reg [7:0] y_valid;
  
  wire w_switch, x_switch;
  
  //transpose variables
  wire [7:0] t_clear_out;
  wire [31:0] t4 [7:0];
  wire [7:0] t5;
  wire [31:0] t_y_out [7:0];
  wire [31:0] x_delay_out [7:0];
  wire temp = 1;
  
  blockmem rf_0(clk, enable, reset, re_0, we_0, in_data, size, chunk_read_0, clear_in_0, chunk_write_0, y_valid, switch_0, rp_0, wp_0, out_data_0);
  blockmem rf_1(clk, enable, reset, re_1, we_1, in_data, size, chunk_read_1, clear_in_1, chunk_write_1, y_valid, switch_1, rp_1, wp_1, out_data_1);
  blockmem rf_2(clk, enable, reset, re_2, we_2, in_data, size, chunk_read_2, clear_in_2, chunk_write_2, y_valid, switch_2, rp_2, wp_2, out_data_2);
  blockmem rf_3(clk, enable, reset, re_3, we_3, in_data, size, chunk_read_3, clear_in_3, chunk_write_3, y_valid, switch_3, rp_3, wp_3, out_data_3);

  genvar i;
  generate
    for(i=0;i<8;i=i+1) begin
      assign w_in[i] = opcode == 1 ? op_b[3:2] == 0 ? chunk_read_0[i] : op_b[3:2] == 1 ? chunk_read_1[i] : op_b[3:2] == 2 ? chunk_read_2[i] : op_b[3:2] == 3 ? chunk_read_3[i] : 0 : 0;
      assign x_in[i] = opcode == 1 ? op_a[3:2] == 0 ? chunk_read_0[i] : op_a[3:2] == 1 ? chunk_read_1[i] : op_a[3:2] == 2 ? chunk_read_2[i] : op_a[3:2] == 3 ? chunk_read_3[i] : 0 : 0;

      assign chunk_write_0[i] = opcode == 1 ? op_c[3:2] == 0 ? y_out[i] : op_e[3:2] == 0 ? {31'b0, b_out[i]} : 0 : 0;
      assign chunk_write_1[i] = opcode == 1 ? op_c[3:2] == 1 ? y_out[i] : op_e[3:2] == 1 ? {31'b0, b_out[i]} : 0 : 0;
      assign chunk_write_2[i] = opcode == 1 ? op_c[3:2] == 2 ? y_out[i] : op_e[3:2] == 2 ? {31'b0, b_out[i]} : 0 : 0;
      assign chunk_write_3[i] = opcode == 1 ? op_c[3:2] == 3 ? y_out[i] : op_e[3:2] == 3 ? {31'b0, b_out[i]} : 0 : 0;
    end
  endgenerate

  assign switch_0 = opcode == 1 ? op_a[3:2] == 0 ? x_switch : op_b[3:2] == 0 ? w_switch : 0 : 0;
  assign switch_1 = opcode == 1 ? op_a[3:2] == 1 ? x_switch : op_b[3:2] == 1 ? w_switch : 0 : 0;
  assign switch_2 = opcode == 1 ? op_a[3:2] == 2 ? x_switch : op_b[3:2] == 2 ? w_switch : 0 : 0;
  assign switch_3 = opcode == 1 ? op_a[3:2] == 3 ? x_switch : op_b[3:2] == 3 ? w_switch : 0 : 0;

  assign clear_in = opcode == 1 ? op_a[3:2] == 0 ? clear_in_0 : op_a[3:2] == 1 ? clear_in_1 : op_a[3:2] == 2 ? clear_in_2 : clear_in_3 : 0;
 
  m8x8 mult(clk, reset, enable, en, op_d, w_in, x_in, clear_in, y_out, clear_out, b_out);

  assign out_data = opcode == 3 ? op_a[3] == 0 ? out_data_0 : op_a[3] == 1 ? out_data_1 : 0 : 0;

  // filter out the upper edge of <opcode == 1> and persist only when memory is still shifting data out.
  assign en = opcode == 1 && !old_en || ind_wc > 0 || ind_wl > 0 || ind_xl > 0;
  
  assign w_switch = ind_wc == size[5:0];
  assign x_switch = w_switch && ind_wl == size[8:6];

  always @(posedge clk) begin
    if(reset) begin // reset behavior: clear all registers
        ind_wc <= 0;
        ind_wl <= 0;
        ind_xl <= 0;
      	y_valid <= 0;
      	old_en <= 0;
    end else if(enable) begin
      if(en) begin
        // count the duration for memory to shift data out
        ind_wc <= w_switch ? 0 : ind_wc + 1;
        ind_wl <= w_switch ? ind_wl == size[8:6] ? 0 : ind_wl + 1 : ind_wl; 
        ind_xl <= x_switch ? ind_xl == size[8:6] ? 0 : ind_xl + 1 : ind_xl;
      end
      // delay <clear_out> for writing memory in bulk
      y_valid <= clear_out;
      // delay <op_code == 1>
      old_en <= opcode == 1;
    end
  end
  
endmodule