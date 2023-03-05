/*
Author: Arthur Wang, Ian Wu
Creation Date: Nov 14 
Last Modified: Dec 8

this is a memory slice that has 1 input port and 1 output port
basic wires:
->  clk: clock
->  enable: global enable
      This is not read enable or write enable.
      If enable is low, it should do nothing.
->  reset: global reset
      This only resets the states of this memory.
      It does not reset memory content
->  size: length of one line

Read-related wires:
->  read_mode: read mode, see below
->  read_page_line: read address
<-  out_data: data output port
<-  next_read_mode: <read_mode> for next memory block to use
<-  read_finish: ast cycle of read of current line
      Only in read mode 1 & 3
<-  last_read: last cycle of read of current line
      Only in read mode 2
<-  next_read_page_line: <read_page_line> for next memory block to use

Write-related wires:
->  write_mode: see below for meaning of each wire
->  in_data: data to save to memory
->  write_page_line: address to write
<-  last_write: signaling the last writing cycle, on DURING the last write
<-  next_write_page_line: <write_page_line>, delayed if in mode 2

Address: The register file has 2048 words. It could be seen as 4x8x64
It means that one register has 4 pages, each page has 8 lines, and each line has 64 elements.
page address: 2 bits
line address: 3 bits
cell address: 6 bits
read_index/write_index: {page address, line address, cell address}, 11 bits in total
read_page_line/write_page_line: {page address, line address}, 5 bits in total
read_cell/write_cell: {cell address}, 6 bits in total


Write Mode: [3:2]
0: Idle
1: Write data serially. The number of cycle for "1" MUST be a multiple of <size>.
2: disabled parallel
3: Write 8 consecutive value. This is primarily for saving results from the 8x8 mult.
    It MUST be on for only 1 cycle, and then for the rest of the time it MUST be "0".

[1:0]
0: idle
1: overwrite
2: hadamard
3: add


Read Mode:
0: Idle
1: Read data in bulk
2: Read data serially

*/
module memory(
  input clk,
  input enable,
  input reset,
  input [3:0] write_mode,
  input [1:0] read_mode,
  input [31:0] in_data,
  input [5:0] size,
  input [4:0] read_page_line,
  input [4:0] write_page_line,
  output [31:0] out_data,
  output [1:0] next_read_mode,
  output reg read_finish,
  output reg last_write,
  output reg last_read,
  output [4:0] next_read_page_line,
  output [4:0] next_write_page_line
);

  reg [5:0] read_cell; // counter for read index
  reg [5:0] write_cell; // counter for write index
  reg [3:0] cont_write; // continuation countdown for Write Mode 2
  reg [4:0] delay_write_page_line; // delayed version of write_page_line
  reg [4:0] delay_read_page_line; // delayed version of read_page_line
  reg delay_bulk_we;
  (* ram_style = "block" *) reg [31:0] data [2047:0]; // BRAM
  reg [1:0] delay_read_mode;
  
  wire [10:0] read_index = read_mode == 3 ? {read_page_line[4:3], read_cell[5:3], read_page_line[2:0], read_cell[2:0]} : {read_page_line, read_cell}; // actual read index
  wire [10:0] write_index = {write_page_line, write_cell}; // actual write index

  assign next_write_page_line = delay_bulk_we ? delay_write_page_line : write_page_line;
  assign next_read_page_line = read_mode == 2 ? read_page_line : delay_read_page_line;
  assign next_read_mode = read_mode == 2 ? 2 : delay_read_mode[0] ? delay_read_mode : 0;

  wire bulk_we = |write_mode[1:0] && write_mode[3] && (write_mode[2] || cont_write > 0);
  wire write_enable = |write_mode[1:0] && (write_mode[2] || cont_write > 0);

  reg [31:0] delay_write_value;
  reg [10:0] delay_write_index;

  wire [31:0] sumof;
  wire [31:0] in_a;

  reg [31:0] data_read;
  reg [2:0] delay_write_mode; 

  wire [10:0] true_write_index = delay_write_mode[1] ? delay_write_index : write_index;

  adder s0(data_read, delay_write_value, sumof);

  assign out_data = write_mode[1] || delay_read_mode == 0 ? 32'b0 : data_read;

  always @(posedge clk) begin
    if(reset) begin // reset behavior
      data_read <= 0;
      read_finish <= 0;
      last_write <= 0;
      last_read <= 0;
      delay_read_page_line <= 0;
      delay_write_page_line <= 0;
      read_cell <= 0;
      write_cell <= 0;
      cont_write <= 0;
      delay_read_mode <= 0;
      delay_write_value <= 0;
      delay_write_index <= 0;
      delay_write_mode <= 0;
    end else if(enable) begin
      // increase read counter / write counter / continuation countdown if enabled
      read_cell <= read_mode > 0 ? read_cell == size ? 0 : read_cell + 1 : read_cell;
      write_cell <= write_enable ? write_cell == size ? 0 : write_cell + 1 : write_cell;
      cont_write <= |write_mode[1:0] && write_mode[3] && write_mode[2] ? 7 : cont_write > 0 ? cont_write - 1 : 0;
      
      data_read <= data[write_mode[1] ? write_index : read_index];
      // perform read operation
      if(write_mode[1]) begin
        delay_write_value <= in_data;
        delay_write_index <= write_index;
      end
      // perform write operation
      if(write_enable && !write_mode[1] || &delay_write_mode[2:1]) begin
        data[true_write_index] <= &delay_write_mode[2:1] ? delay_write_mode[0] ? sumof : data_read == 1 ? delay_write_value : 0 : in_data;
      end
      // delayed version of inputs
      delay_read_mode <= read_mode;
      delay_read_page_line <= read_page_line;
      delay_write_page_line <= bulk_we ? write_page_line : 0;
      delay_bulk_we <= bulk_we;
      delay_write_mode <= {write_enable, write_mode[1:0]};
      // only in read mode 1 & 3
      read_finish <= read_mode[0] && read_cell == size;
      // ending flags. for last_write, use size - 1 because it has extra delay
      last_write <= |write_mode[1:0] && write_mode[3:2] == 1 && write_cell == size - 1 || bulk_we && write_cell == size - 1;
      last_read <= read_mode == 2 && read_cell == size - 1;
    end
  end
  
endmodule