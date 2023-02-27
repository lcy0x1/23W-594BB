module quest_for_out();

integer  i;
reg clk;
wire a = clk;
logic b = clk;
reg c = clk;


initial begin
    clk = 0;
    #4 $finish;
end

always #1 clk = !clk;

initial begin
    $dumpfile("dump.vcd");
    $dumpvars;
    b = clk;
    c = clk;
    #10
    $finish;
end

endmodule