`timescale 1ns/100ps

module testbench();
  
    parameter CLK = 10;

    reg clk;
    reg reset;
    reg s1, s2, s3;

    wire spike;
    wire unsigned [3:0] Vi;
  
    wrapper main(clk, reset, s1, s2, s3, spike, Vi);

`ifdef SET1
    wire [34:0] data1 = 35'b11111001111100111110011111001111100;
    wire [34:0] data2 = 35'b11111001111100111110011111001111100;
    wire [34:0] data3 = 35'b11111001111100111110011111001111100;
`endif

`ifdef SET2
    wire [34:0] data1 = 35'b11111001111100111110011111001111100;
    wire [34:0] data2 = 35'b11111001111100111110011111001111100;
    wire [34:0] data3 = 35'b00000110000011000001100000110000011;
`endif

`ifdef SET3
    wire [34:0] data1 = 35'b00000110000011000001100000110000011;
    wire [34:0] data2 = 35'b00000110000011000001100000110000011;
    wire [34:0] data3 = 35'b11111001111100111110011111001111100;
`endif

`ifdef SET4
    wire [34:0] data1 = 35'b00000000001111100000000001111100000;
    wire [34:0] data2 = 35'b00000111110000000000111110000000000;
    wire [34:0] data3 = 35'b11111000000000011111000000000011111;
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
    for(int i=34; i>=0; i--) begin
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
    #(CLK);
    #(CLK);
    #(CLK);
    #(CLK);
    $finish;
  end

endmodule