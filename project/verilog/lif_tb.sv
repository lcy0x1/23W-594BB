

`timescale 1 ns/10 ps  // time-unit = 1 ns, precision = 10 ps
module lif_tb;
    localparam period = 20;  
    reg clk,
    reg rstn,
    reg `SIG_V spike_in,
    wire spike_out
    lif UUT (clk,rstn,`SIG_V spike_in,spike_out);

// clock period = 2 ns
always 
begin
    clk = 1'b1; 
    #20; 

    clk = 1'b0;
    #20; 
end

always @(posedge clk)
begin
    // values for
    a = 0;
    b = 0;
    #period; // wait for period

    if(sum != 0 || carry != 0)  
        $display("test failed for input combination 00");

    $stop;   // end of simulation
end
endmodule