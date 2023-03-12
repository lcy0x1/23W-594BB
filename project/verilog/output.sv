`timescale 1ns/1ps
`define DEF_OUTPUT_SIZE 10
`define DEF_OUTPUT_WIDTH 4
`define DEF_COUNTER_WIDTH 8
`define DEF_V_SIZE 4

/* This module counts the spikes from the network till the valid bit goes low.
 * Then it will output the channel with the highest spike count
*/
module output_layer #(
    parameter V_SIZE = `DEF_V_SIZE,
    parameter OUTPUT_SIZE = `DEF_OUTPUT_SIZE,
    parameter OUTPUT_WIDTH = `DEF_OUTPUT_WIDTH,
    parameter COUNTER_WIDTH = `DEF_COUNTER_WIDTH
    ) (
    input clk,
    input rstn,
    input [OUTPUT_SIZE-1:0] spike,
    input in_valid,
    output logic [OUTPUT_WIDTH-1:0] result,
    output reg out_valid
);

reg [COUNTER_WIDTH-1:0] counter [OUTPUT_SIZE-1:0];
reg valid_prev;

always_comb begin
    integer i;
    integer max;
    result = 0;
    max = 0;
    for (i = 0; i < OUTPUT_SIZE; i++) begin
        if (counter[i] > max) begin
            max = counter[i];
            result = i;
        end
    end
end

always @(posedge clk) begin
    integer i;
    if (rstn==0) begin
        out_valid <= 0;
        valid_prev <= 0;
        for (i = 0; i < OUTPUT_SIZE; i++) begin
            counter[i] <= 0;
        end
    end else begin
        for (i = 0; i<OUTPUT_SIZE; i++) begin
            if (spike[i] && in_valid) begin
                counter[i] <= counter[i] + 1;
            end else begin
                counter[i] <= counter[i];
            end
        end
        if (valid_prev && !in_valid) begin
            out_valid <= 1;
            valid_prev <= in_valid;
        end else begin
            out_valid <= 0;
            valid_prev <= in_valid;
        end
    end
end

endmodule