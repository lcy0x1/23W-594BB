/* Wrapper module
 * input: 13 bit of command or control, 19 bit of data
*/
module wrapper(
    input clk,
    input rstn,
    input [1:0] opcode_i, // 0 = idle, 1 = write, 2 = info
    input [18:0] data_in,
    output [3:0] result_o,
    output [1:0] status_o // 0 = idle, 1 = busy, 2 = idle + result
);

wire write_en = opcode_i == 1;
wire settled;
wire [18:0] write_data = write_en ? data_in : 0;
wire [18:0] net_i;
wire [9:0] net_o;

reg [7:0] read_timer;
reg [7:0] count_timer;

wire read_en = read_timer > 0;
wire count_en = count_timer > 0;

assign status_o = count_en ? 1 : settled ? 2 : 0;

memory #(19, 128, 7) mem(clk, rstn, read_en, write_en, write_data, net_i);
generated #(8) network(clk, rstn, net_i, net_o);
output_layer #(8, 10, 4, 7) aggregate(clk, rstn, net_o, count_en, result_o, settled);

always @(posedge clk) begin
    if(!rstn) begin
        read_timer <= 0;
        count_timer <= 0;
    end else begin
        if (!count_en && opcode_i == 2) begin
            read_timer <= data_in[7:0];
            count_timer <= data_in[17:10] + data_in[7:0];
        end else begin
            read_timer <= |read_timer ? read_timer - 1 : 0;
            count_timer <= |count_timer ? count_timer - 1 : 0;
        end
    end
end

endmodule