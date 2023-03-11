`define DEF_ROW 19
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
    input [ROW-1:0] data_in,
    output reg [ROW-1:0] data_out
);

reg [LOG_WIDTH-1:0] in_addr, out_addr;

reg [ROW-1:0] data [0:WIDTH-1];

always @(posedge clk) begin
    if (!rstn) begin
        in_addr <= 0;
        out_addr <= 0;
        data_out <= 0;
    end else if (we) begin
        data[in_addr] <= data_in;
        out_addr <= 0;
        in_addr <= in_addr + 1;
        data_out <= 0;
    end else if (en) begin
        data_out <= data[out_addr];
        out_addr <= out_addr + 1;
        in_addr <= 0;
    end else begin
        data_out <= 0;
        in_addr <= 0;
        out_addr <= 0;
    end
end

endmodule