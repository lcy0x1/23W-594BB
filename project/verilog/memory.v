`define DEF_ROW 16
`define DEF_WIDTH 128
`define DEF_LOG_WIDTH 7

module memory #(
    parameter ROW = `DEF_ROW, 
    parameter WIDTH = `DEF_WIDTH,
    parameter LOG_WIDTH = `DEF_LOG_WIDTH
) (
    input clk,
    input rstn,
    input en,
    input we,
    input [DEF_WORD-1:0] data_in,
    output [DEF_ROW-1:0] data_out
);

reg [LOG_WIDTH-1:0] in_addr, out_addr;

reg [WORD-1:0] data [0:WIDTH-1]

always (posedge clk) begin
    if (!rstn) begin
        in_addr <= 0;
        out_addr <= 0;
    end else if (en) begin
        
    end else if (we) begin

    end
end


endmodule