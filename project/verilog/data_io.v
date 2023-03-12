module data_io(
    input clk,
    input rstn,
    input [15:0] data_in,
    output [15:0] data_out
);

wire opcode_i = data_in[15:14];
wire parity_i = data_in[11:10]; // high bit: write enable, low bit: 1/0 = msb/lsb
wire indata_i = data_in[9:0];

reg [1:0] data_parity;
reg [18:0] acc_data;
reg data_valid;

wire [1:0] opcode = data_valid ? opcode_i : 0;

wire [6:0] result;
wire [1:0] status;

assign data_out = {7'b0, status, result};

wrapper uut(clk, rstn, opcode, acc_data, result, status);

always @(posedge clk) begin
    if (!rstn) begin
        data_parity <= 0;
        acc_data <= 0;
        data_valid <= 0;
    end else begin
        if(data_parity != parity_i) begin
            data_parity <= parity_i;
            if(parity_i[1]) begin
                if(parity_i[0]) begin
                    acc_data[9:0] <= indata_i;
                    data_valid <= 0;
                end else begin
                    acc_data[18:10] <= indata_i[8:0];
                    data_valid <= 1;
                end 
            end else begin
                data_valid <= 0;
            end
        end else begin
            data_valid <= 0;
        end
    end
end

endmodule