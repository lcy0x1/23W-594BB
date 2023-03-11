/* Wrapper module
 * input: 13 bit of command or control, 19 bit of data
*/
module wrapper(
    input clk,
    input rstn,
    input [31:0] data_in,
    output [31:0] data_out
);

wire read_en;
wire write_en;
wire settled;
wire [18:0] write_data = write_en ? data_in[18:0] : 0;
wire [18:0] net_i;
wire [9:0] net_o;

memory #(19, 128, 7) mem(clk, rstn, read_en, write_en, write_data, net_i);
generated #(9) network(clk, rstn, net_i, net_o);
output_layer #(9, 10, 128, 7) aggregate(clk, rstn, net_o, read_en, settled);

endmodule