```
brew install icarus-verilog
brew install node
iverilog '-Wall' '-g2012' main.v testbench.sv  && vvp a.out
```