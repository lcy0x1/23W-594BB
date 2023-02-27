`timescale 1ns/100ps

module testbench();
  
    parameter CLK = 10;

    reg clk;
    reg reset;
    reg s1, s2, s3;
  
    wrapper main(clk, reset, s1, s2, s3);

`ifdef SET1
    wire [39:0] data1 = 40'b0010010010010010010010010010010010010010;
    wire [39:0] data2 = 40'b0000011111000000000011111000000000000000;
    wire [39:0] data3 = 40'b1111100000000001111100000000001111100000;
`endif

`ifdef SET2
    wire [39:0] data1 = 40'b1100000110000011000001100000110000011000;
    wire [39:0] data2 = 40'b0000011000001100000110000011000001100000;
    wire [39:0] data3 = 40'b1111100000000001111100000000001111100000;
`endif

`ifdef SET3
    wire [39:0] data1 = 40'b0000011000001100000110000011000001100000;
    wire [39:0] data2 = 40'b1100000000110000000011000000001100000000;
    wire [39:0] data3 = 40'b0000011000001100000110000011000001100000;
`endif

`ifdef SET4
    wire [39:0] data1 = 40'b1010101010101010101010101010101010101010;
    wire [39:0] data2 = 40'b0010010010010010010010010010010010010010;
    wire [39:0] data3 = 40'b0000011000001100000110000011000001100000;
`endif

  initial begin
    #10;
    forever begin
      clk = ~clk;
      #5;
    end
  end

  integer step;
  
  initial begin
    $dumpfile ("dump.vcd");
    $dumpvars;
    clk = 0;
    reset = 0;
    s1 = 0;
    s2 = 0;
    s3 = 0;
    step = 0;
    #(CLK);
    reset = 1;
    #(CLK);
    reset = 0;
    #(CLK);
    #(CLK);
    #(CLK);
    for(int i=39; i>=0; i--) begin
        s1 = data1[i];
        s2 = data2[i];
        s3 = data3[i];
        step = step + 1;
        #(CLK);
    end
    step = step + 1;
    s1 = 0;
    s2 = 0;
    s3 = 0;
    #(CLK);
    step = step + 1;
    #(CLK);
    #(CLK);
    #(CLK);
    #(CLK);
    $finish;
  end

endmodule