module data_io(
    input clk,
    input rstn,
    input [22:0] data_in,
    output [5:0] data_out
);

wire [1:0] opcode_i = data_in[22:21];
wire [1:0] parity_i = data_in[20:19]; // high bit: write enable, low bit: 1/0 = msb/lsb
wire [18:0] indata_i = data_in[18:0];

reg [1:0] data_parity;
reg [18:0] acc_data;
reg data_valid;

wire [3:0] result;
wire [1:0] status;

assign data_out = {status, result};

wrapper uut(clk, rstn, opcode_i, acc_data, result, status);

always @(posedge clk) begin
    if (!rstn) begin
        data_parity <= 0;
        acc_data <= 0;
        data_valid <= 0;
    end else begin
        if(data_parity != parity_i) begin
            data_parity <= parity_i;
            if(parity_i[1]) begin
                acc_data <= indata_i;
                data_valid <= 1;
            end else begin
                data_valid <= 0;
            end
        end else begin
            data_valid <= 0;
        end
    end
end

endmodule