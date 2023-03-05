module my_fp_mult(
    input [31:0] a,
    input [31:0] b,
    output [31:0] y
);

    wire AS = a[31];
    wire [7:0] AE = a[30:23];
    wire [22:0] AM = a[22:0];
    wire BS = b[31];
    wire [7:0] BE = b[30:23];
    wire [22:0] BM = b[22:0];
  	wire [47:0] prod = {1'b1,AM}*{1'b1,BM};
    wire [8:0] sum_exp = AE + BE + prod[47];
    wire [22:0] y_mts = prod[47] ? prod[46:24] : prod[45:23];
    wire [8:0] raw_y_exp = sum_exp - 9'b001111111;
    wire [7:0] y_exp = raw_y_exp[7:0];
    wire under = ~|sum_exp[8:7] && ~&sum_exp[6:0];
    wire over = !under && raw_y_exp[8];

    wire azm = ~|AM;
    wire bzm = ~|BM;
    wire a_zero = ~|AE & azm;
    wire b_zero = ~|BE & bzm;
    wire a_sp = &AE;
    wire b_sp = &BE;
    wire a_inf = a_sp & azm;
    wire b_inf = b_sp & bzm;
    wire a_nan = a_sp & !azm;
    wire b_nan = b_sp & !bzm;

    wire y_nan = a_nan || b_nan || a_inf && b_zero || a_zero && b_inf;
    wire y_inf = !y_nan && (a_inf || b_inf || over);
    wire y_zero = !y_nan && (a_zero || b_zero || under);

    wire YS = AS ^ BS;
  	wire [7:0] YE = y_zero ? 8'h00 : y_nan || y_inf ? 8'hFF : y_exp;
  	wire [22:0] YM = y_zero || y_inf ? 23'b0 : y_nan ? 23'b1 : y_mts;
    assign y = {YS, YE, YM};

endmodule