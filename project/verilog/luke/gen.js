const x_arr = [];
const w_arr = [];
const w2_arr = [];
const w3_arr = [];

const dy_arr = [];

let seed = 1234567;
for (let i = 0; i < 64; i++) {
    for (let j = 0; j < 64; j++) {
        if (!w_arr[i]) w_arr[i] = [];
        if (!x_arr[i]) x_arr[i] = [];
        if (!w2_arr[i]) w2_arr[i] = [];
        if (!w3_arr[i]) w3_arr[i] = [];
        if (!dy_arr[i]) dy_arr[i] = [];
        w_arr[i][j] = seed % 131072 - 65536;
        seed = seed * 107 % 100000007;
        x_arr[i][j] = seed % 131072 - 65536;
        seed = seed * 107 % 100000007;
        w2_arr[i][j] = seed % 131072 - 65536;
        seed = seed * 107 % 100000007;
        w3_arr[i][j] = seed % 131072 - 65536;
        seed = seed * 107 % 100000007;
        dy_arr[i][j] = seed % 131072 - 65536;
        seed = seed * 107 % 100000007;
    }
}

let str0 = '';
str0 += `in_data = 256;\n#(CLK);\nin_data = 32'h02;\n#(CLK);\n`;
for (let i1 = 0; i1 < 2; i1++) {
    for (let i0 = 0; i0 < 8; i0++) {
        for (let j = 0; j < 16; j++) {
            str0 += `in_data = ${w_arr[i1 * 8 + i0][j]};\n`;
            str0 += '#(CLK);\n';
        }
    }
}

str0 += `in_data = 256;\n#(CLK);\nin_data = 32'h42;\n#(CLK);\n`;
for (let i = 0; i < 16; i++) {
    for (let j = 0; j < 16; j++) {
        str0 += `in_data = ${w2_arr[i][j]};\n`;
        str0 += '#(CLK);\n';
    }
}

str0 += `in_data = 256;\n#(CLK);\nin_data = 32'h12;\n#(CLK);\n`;
for (let i = 0; i < 16; i++) {
    for (let j = 0; j < 16; j++) {
        str0 += `in_data = ${w3_arr[i][j]};\n`;
        str0 += '#(CLK);\n';
    }
}

str0 += `in_data = 256;\n#(CLK);\nin_data = 32'h52;\n#(CLK);\n`;
for (let i = 0; i < 16; i++) {
    for (let j = 0; j < 16; j++) {
        str0 += `in_data = ${x_arr[i][j]};\n`;
        str0 += '#(CLK);\n';
    }
}


// multiplication of W1 and X is W1X'
// here it means transpose X, W1=#0, X=#5, answer is W1*X
//A1 = W1X
str0 += `in_data = 128;\n#(CLK);\nin_data = 32'h00962501;\n#(CLK*129);\n`;
str0 += `in_data = 128;\n#(CLK);\nin_data = 32'h00c66241;\n#(CLK*129);\n`;


// 0: unused
// 0: overwrite
// 8: relu derivative: B3
// 6: transpose second operand, use ReLU
// d: target: Y
// 6: second operand: A2
// 1: first operand: W3
// 1: opcode = 1
// Y = W3 * A2
str0 += `in_data = 128;\n#(CLK);\nin_data = 32'h0086d611;\n#(CLK*129);\n`;



str0 += `in_data = 256;\n#(CLK);\nin_data = 32'h182;\n#(CLK);\n`;
for (let i = 0; i < 16; i++) {
    for (let j = 0; j < 16; j++) {
        str0 += `in_data = ${dy_arr[i][j]};\n`;
        str0 += '#(CLK);\n';
    }
}


// Operation 5
// 0: unused
// 1: hadamard product
// 0: no relu derivative
// c: transpose both, no relu
// c: target: P2
// 8: second operand: P3
// 1: first operand: W3^T
// 1: opcode = 1

str0 += `in_data = 128;\n#(CLK);\nin_data = 32'h010cc811;\n#(CLK*129);\n`;

// Operation 6
// 0: unused
// 2: addition
// 0: no relu derivative
// 0: no transpose, no relu
// 1: target: W3
// 6: second operand: A2^T
// 8: first operand: P3
// 1: opcode = 1
str0 += `in_data = 128;\n#(CLK);\nin_data = 32'h02001681;\n#(CLK*129);\n`;

// Operation 7
// 0: unused
// 1: hadamard product
// 0: no relu derivative
// c: transpose both, no relu
// 9: target: P1
// c: second operand: P2
// 0: first operand: W2^T
// 1: opcode = 1
str0 += `in_data = 128;\n#(CLK);\nin_data = 32'h010c9c01;\n#(CLK*129);\n`;

// Operation 8
// 0: unused
// 2: addition
// 0: no relu derivative
// 0: no transpose, no relu
// 4: target: w2
// 2: second operand: A1^T
// c: first operand: P2
// 1: opcode = 1
str0 += `in_data = 128;\n#(CLK);\nin_data = 32'h020042c1;\n#(CLK*129);\n`;

// Operation 9
// 0: unused
// 2: addition
// 0: no relu derivative
// 0: no transpose, no relu
// 0: target: w1
// 4: second operand: X^T
// 9: first operand: P1
// 1: opcode = 1
str0 += `in_data = 128;\n#(CLK);\nin_data = 32'h02000491;\n#(CLK*129);\n`;


const fs = require('fs');
fs.writeFileSync('gen.txt', str0);