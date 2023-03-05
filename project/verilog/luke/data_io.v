`include "data_interface.v"
/*
Author: Arthur Wang
Create Data: Oct 31

Bridge between verilog modules and DMA

*/

module data_io(
    // clock and reset ports - since both AXI buses are on the same clock in the block diagram, we only need one of each
    input wire  M_AXIS_ACLK,
    input wire  M_AXIS_ARESETN,
    input wire  S_AXIS_ACLK,
	input wire  S_AXIS_ARESETN,
    
    //master AXI interface - sends data back to DRAM
    output wire  M_AXIS_TVALID, // Master Stream Ports. TVALID indicates that the master is driving a valid transfer, A transfer takes place when both TVALID and TREADY are asserted. 
    output wire [31:0] M_AXIS_TDATA, // TDATA is the primary payload that is used to provide the data that is passing across the interface from the master.
    output wire [3:0] M_AXIS_TKEEP, // 
    output reg  M_AXIS_TLAST, // TLAST indicates the boundary of a packet.
    input wire  M_AXIS_TREADY, // TREADY indicates that the slave can accept a transfer in the current cycle.
    
    //slave AXI interface - recieves data from DRAM
    output wire  S_AXIS_TREADY, // Ready to accept data in
    input wire [31:0] S_AXIS_TDATA, // Data in
    input wire [3:0] S_AXIS_TKEEP, // almost always high - indicates that data bytes are not null
    input wire  S_AXIS_TLAST, // Indicates boundary of last packet
    input wire  S_AXIS_TVALID // Data is in valid
    );
    
    //add every pair of elements.  Looks like a convolution of a [1,1] filter, with stride 2
    
    reg in_state;
    
    reg [1:0] out_state;
    //00: empty queue
    //01: output available
    //10: output full
    //11: Invalid
    
    reg [15:0] in_buffer;
    reg in_enable;
    
    reg [15:0] in_count;
    reg [15:0] out_count;
    wire [15:0] data_out;
    wire [15:0] wr_out_count;
    wire wr_outc_valid;
    
    reg [15:0] databuf [1:0]; //store each pair of data
    wire TX, RX;
    assign TX = M_AXIS_TREADY && M_AXIS_TVALID; //internal flag: we transmitted a word to output stream
    assign RX = S_AXIS_TREADY && S_AXIS_TVALID; //internal flag: we recieved a word from input stream
    assign M_AXIS_TVALID = out_state == 1 || out_state == 2 || M_AXIS_TLAST; //we want to transmit if we are in state 2, or if we have reached the end of a packet
    assign M_AXIS_TKEEP = (out_state == 1 || out_state == 2) ? 15 : 0; //if we are in state 2, then the data values are a sum that we want to save and should be labeled as such
    
    assign S_AXIS_TREADY = (out_state < 2) && M_AXIS_TREADY && itf_ready; //ready to process a new bit of data as long as output FIFO is ready and interface is ready
    assign M_AXIS_TDATA = 
            out_state == 1 ? databuf[0] : 
            out_state == 2 ? databuf[1] : 0;
    
    wire enable = out_state < 2 && in_enable && (RX || in_count == 0 || ~itf_ready);
    wire itf_y_valid;
    wire y_valid = enable && itf_y_valid;
    
    data_interface itf(S_AXIS_ACLK, !S_AXIS_ARESETN, in_buffer, enable, data_out, itf_y_valid, wr_out_count, wr_outc_valid, itf_ready);
    
    always @(posedge S_AXIS_ACLK) begin
        if (!S_AXIS_ARESETN) begin
            out_state <= 0;
            databuf[0] <= 0;
            databuf[0] <= 0;
            in_count <= 0;
            out_count <= 0;
            M_AXIS_TLAST <= 0;
            
            in_state <= 0;
            in_buffer <= 0;
            in_enable <= 0;
        end
        else begin
            //state update
            if (~M_AXIS_TLAST) begin
                if (in_state == 0) in_state <= RX;
                if (out_state == 0) out_state <= in_state && y_valid;
                if (out_state == 1) out_state <= y_valid ? TX ? 1 : 2 : TX ? 0 : 1;
                if (out_state == 2) out_state <= TX ? 1 : 2;
                if(out_count == 0 && wr_outc_valid) out_count <= wr_out_count;
                if(out_count > 0 && TX) out_count <= out_count - 1;
            end
            else begin //if a TLAST has been recieved, then the last thing to do is send a tlast onward to the output FIFO 
                out_state <= 3;
                databuf[0] <= 0;
                databuf[1] <= 0;
            end
            
            //edge updates
            if (RX) begin
                if(~in_state) begin
                    in_count <= S_AXIS_TDATA;
                end else begin
                    if(in_count > 0) begin
                        in_count <= in_count - 1;
                    end
                    in_buffer <= S_AXIS_TDATA;
                    in_enable <= in_count > 0;
                end
            end
            
            if(y_valid) begin
                databuf[0] <= data_out;
                databuf[1] <= databuf[0];
            end
            M_AXIS_TLAST <= M_AXIS_TLAST ? 0 : out_count == 2;
        end
    
    end
endmodule