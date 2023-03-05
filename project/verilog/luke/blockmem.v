`include "memory.v"

/*
Author: Arthur Wang
Creation Date: Nov 14 
Last Modified: Dec 8

Spec:
This block has 8 slices, each slice has 4 pages, each page has 8 lines, each line has 64 cells.
Each slice is currently taking 8KB from a single BRAM.
Thus, we can increase page size up to 16 pages to take up full 32KB BRAM
We can also have at most 4 blocks so that it takes 32 BRAMs

basic wires:
->  clk: clock
->  enable: global enable. If it is low, nothing should be done
->  reset: global reset
      It only reset states. It does not clear memory 
->  size: size of data

Read related flags:
->  read_mode: see below
->  page_read: page address to read
->  switch: signal to switch line
<-  x_out: output bus
<-  clear_out: For matrix multiplier unit. 
      It is the last data on the respective line

Write related flags:
->  write mode: see below
->  in_data: input port for serial write mode
->  bus_in: input port for bulk write mode
->  bus_valid: input valid port for bulk write mode
->  page_write: page address to write

Write Mode [1:0]:
0: idle
1: normal
2: hadamard
3: addition

Write Mode [2]:
0: serial write, one data at a time. The size of input MUST match <size>
    Implementation: The <memory> block will iterate through every cell on a line.
    When the last value is written, one of the <lw> flags will be on and <write_slice_ind> will increment
    so that it will write to the next slice. After the last slice is written,
    the <lw[7]> flag will be on and the <waddr> will increase, so that is is writing the next line
    of each slices. Basically for(line) for(slice) for(cell)
1: bulk write, 8 data at a time. 
    It should be on for the entire duration of write.
    To be safe, it should be on before operation occurs and off after all operation ceases.
    <bus_valid> signal is on for only 1 cycle at a time, signaling the beginning of 8 data.
    It should not be on for the entire duration of 8 valid data.

Read Mode:
0: idle
1: bulk read. It should be one for the entire span of reading process.
2: serial read.
3: transposed read

Addressing:
size[5:0] tells the number of data per line
size[8:6] tells the number of line per page
page_read and page_write tells the page address of read and write operation

*/

module blockmem(
  input clk,
  input enable,
  input reset,
  input [1:0] read_mode,
  input [2:0] write_mode,
  input [31:0] in_data,
  input [8:0] size,
  output [31:0] x_out [7:0],
  output [7:0] clear_out,
  input [31:0] bus_in [7:0],
  input [7:0] bus_valid,
  input switch,
  input [1:0] page_read,
  input [1:0] page_write,
  output [31:0] out_data
);

  // read related variables
  wire [1:0] mem_en [7:0]; // read enable, on once, effective for <size> clocks
  reg [2:0] read_slice_ind; // index of currently active reading slice. Only for read mode 2
  reg [2:0] raddr; // address of line to read
  wire [4:0] mid_raddr [7:0]; // cached read address
  wire [7:0] lr;
  reg [2:0] delay_read_slice_ind;

  // write related variables
  wire [7:0] lw; // flags for last write, for updating <ind>
  reg [2:0] write_slice_ind; // index of currently active writing slice. Only for Write Mode 1
  reg [2:0] waddr; // address of line to write
  wire [4:0] mid_waddr [7:0]; // cached waddr

  // slice write modes:
  // write mode == 2 && bus_valid[i] -> start bulk write, send flag 2
  // write mode == 1 && ind == i -> enable serial write, set flag 1

  wire [3:0] wms [7:0];

  assign wms[0] = |write_mode[1:0] ? { write_mode[2] ? {1'b1, bus_valid[0]} : {1'b0, write_slice_ind == 0} , write_mode[1:0] } : 3'b000;
  assign wms[1] = |write_mode[1:0] ? { write_mode[2] ? {1'b1, bus_valid[1]} : {1'b0, write_slice_ind == 1} , write_mode[1:0] } : 3'b000;
  assign wms[2] = |write_mode[1:0] ? { write_mode[2] ? {1'b1, bus_valid[2]} : {1'b0, write_slice_ind == 2} , write_mode[1:0] } : 3'b000;
  assign wms[3] = |write_mode[1:0] ? { write_mode[2] ? {1'b1, bus_valid[3]} : {1'b0, write_slice_ind == 3} , write_mode[1:0] } : 3'b000;
  assign wms[4] = |write_mode[1:0] ? { write_mode[2] ? {1'b1, bus_valid[4]} : {1'b0, write_slice_ind == 4} , write_mode[1:0] } : 3'b000;
  assign wms[5] = |write_mode[1:0] ? { write_mode[2] ? {1'b1, bus_valid[5]} : {1'b0, write_slice_ind == 5} , write_mode[1:0] } : 3'b000;
  assign wms[6] = |write_mode[1:0] ? { write_mode[2] ? {1'b1, bus_valid[6]} : {1'b0, write_slice_ind == 6} , write_mode[1:0] } : 3'b000;
  assign wms[7] = |write_mode[1:0] ? { write_mode[2] ? {1'b1, bus_valid[7]} : {1'b0, write_slice_ind == 7} , write_mode[1:0] } : 3'b000;

  memory slice_0(clk, enable, reset, wms[0], read_mode, write_mode[2] ? bus_in[0] : in_data, size[5:0], {page_read, raddr}, {page_write, waddr}, x_out[0], mem_en[0], clear_out[0], lw[0], lr[0], mid_raddr[0], mid_waddr[0]);
  memory slice_1(clk, enable, reset, wms[1], mem_en[0], write_mode[2] ? bus_in[1] : in_data, size[5:0], mid_raddr[0],       mid_waddr[0],        x_out[1], mem_en[1], clear_out[1], lw[1], lr[1], mid_raddr[1], mid_waddr[1]);
  memory slice_2(clk, enable, reset, wms[2], mem_en[1], write_mode[2] ? bus_in[2] : in_data, size[5:0], mid_raddr[1],       mid_waddr[1],        x_out[2], mem_en[2], clear_out[2], lw[2], lr[2], mid_raddr[2], mid_waddr[2]);
  memory slice_3(clk, enable, reset, wms[3], mem_en[2], write_mode[2] ? bus_in[3] : in_data, size[5:0], mid_raddr[2],       mid_waddr[2],        x_out[3], mem_en[3], clear_out[3], lw[3], lr[3], mid_raddr[3], mid_waddr[3]);
  memory slice_4(clk, enable, reset, wms[4], mem_en[3], write_mode[2] ? bus_in[4] : in_data, size[5:0], mid_raddr[3],       mid_waddr[3],        x_out[4], mem_en[4], clear_out[4], lw[4], lr[4], mid_raddr[4], mid_waddr[4]);
  memory slice_5(clk, enable, reset, wms[5], mem_en[4], write_mode[2] ? bus_in[5] : in_data, size[5:0], mid_raddr[4],       mid_waddr[4],        x_out[5], mem_en[5], clear_out[5], lw[5], lr[5], mid_raddr[5], mid_waddr[5]);
  memory slice_6(clk, enable, reset, wms[6], mem_en[5], write_mode[2] ? bus_in[6] : in_data, size[5:0], mid_raddr[5],       mid_waddr[5],        x_out[6], mem_en[6], clear_out[6], lw[6], lr[6], mid_raddr[6], mid_waddr[6]);
  memory slice_7(clk, enable, reset, wms[7], mem_en[6], write_mode[2] ? bus_in[7] : in_data, size[5:0], mid_raddr[6],       mid_waddr[6],        x_out[7], mem_en[7], clear_out[7], lw[7], lr[7], mid_raddr[7], mid_waddr[7]);
  
  assign out_data = read_mode == 2 ? x_out[{1'b0, delay_read_slice_ind}] : 32'b0;

  always @(posedge clk) begin
    if(reset) begin // reset behavior: clear al registers
      read_slice_ind <= 0;
      write_slice_ind <= 0;
      raddr <= 0;
      waddr <= 0;
      delay_read_slice_ind <= 0;
    end else if(enable) begin
      // the waddr update logic is different in write mode 1 and write mode 2
      if(|write_mode[1:0] && !write_mode[2]) begin
        write_slice_ind <= |lw ? write_slice_ind == 7 ? 0 : write_slice_ind + 1 : write_slice_ind;
        waddr <= lw[7] ? waddr == size[8:6] ? 0 : waddr + 1 : waddr;
      end else if(|write_mode[1:0] && write_mode[2]) begin 
        // increase the line index <waddr> on finishing a line, all slices are written together
        waddr <= lw[0] ? waddr == size[8:6] ? 0 : waddr + 1 : waddr;
      end
      if(read_mode[0] == 1) begin
        // read address increment by 1 when <switch> is on
        raddr <= switch ? raddr == size[8:6] ? 0 : raddr + 1 : raddr;
      end else begin
        read_slice_ind <= |lr ? read_slice_ind == 7 ? 0 : read_slice_ind + 1 : read_slice_ind;
        raddr <= |lr && read_slice_ind == 7 ? raddr == size[8:6] ? 0 : raddr + 1 : raddr;
        delay_read_slice_ind <= read_slice_ind;
      end
    end
  end
    
endmodule