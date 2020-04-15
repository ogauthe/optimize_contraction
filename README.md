Exhaustive searche algorithm to find the best contraction sequence of a tensor network.

**The program does not currently implements outer products of two tensors.** It only checks whether it may be necessary and displays a warning in such cases. It processes by contractring tensors two by two over all their shared legs, which is never suboptimal.


# Installation
If you have not already installed Rust, follow the instructions at https://www.rust-lang.org/tools/install.
Clone the repository with
```
git clone https://github.com/gauthe/optimize_contraction.git
```
compile with
```
cargo build --release
```
the optimized executable will be `current_dir/target/release/optimize_contraction`.

# Usage
Write an input file for your own tensor network following the json syntax of `input_sample.json`. Then call the executable with the input file as argument. Without any input, the program will use the sample.
The program checks that:
* the provided tensor network is connected;
* no leg has dimension 0 or 1;
* no tensor has twice the same leg (always optimal to trace over those legs first);
* a given leg appears either once (free leg) or twice (leg to contract) only.

The program prints the best contraction sequence as well as its CPU cost and an upper bound for the memory in the standard output.

# Code generation
The program `generate_py/generate_contraction_code.py` can then be used to generate a memory-optimized python code. It allows formal variables as dimensions, bypass tensordot to reduce the number of tensor copy and delete temporary tensors. Unfortunately the output is not very readable and I advise to review it before inserting it inside real code.

# References
Pfeiffer et al., Phys. Rev. E 90, 033315, https://journals.aps.org/pre/abstract/10.1103/PhysRevE.90.033315  
see also https://github.com/frankschindler/OptimizedTensorContraction/
