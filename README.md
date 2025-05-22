# Aquila Optimizer (AO)

A comprehensive implementation of the Aquila Optimizer algorithm, a novel meta-heuristic optimization algorithm inspired by the hunting behavior of Aquila birds. This project serves as a reproducibility study of the original paper:

> Abualigah, L., Yousri, D., Abd Elaziz, M., Ewees, A.A., Al-Qaness, M.A. and Gandomi, A.H., 2021. Aquila optimizer: a novel meta-heuristic optimization algorithm. Computers & Industrial Engineering, 157, p.107250.

## Authors

This implementation was created by:
- S. Parashar (sp3466)
- K. Sasanapuri (ks2992)
- A. Khatu (ak7665)

as part of CSCI 633: Biologically Inspired Intelligent Systems course project.

## Project Overview

This implementation provides:
- Core Aquila Optimizer algorithm implementation
- Extensive benchmark testing capabilities
- Support for custom objective functions
- Parallel processing for efficient testing
- Comprehensive results analysis

## Project Structure

```
.
├── aquila_optimizer.py          # Core AO algorithm implementation
├── master.py                    # CEC benchmark test runner
├── master_classical.py          # Classical benchmark test runner
├── worker_AO.py                 # Worker for CEC benchmark tests
├── worker_AO_classical.py       # Worker for classical benchmark tests
├── original paper.pdf           # Original research paper
└── outputs/                     # Directory for test results
```

## Features

### 1. Core Algorithm
- Population-based optimization
- Two-phase optimization strategy:
  - Exploration phase (first 2/3 of iterations)
  - Exploitation phase (last 1/3 of iterations)
- Levy flight distribution for enhanced exploration
- Adaptive parameter adjustment

### 2. Benchmark Testing
The implementation supports testing on:
- 29 CEC 2017 benchmark functions
- 9 CEC 2019 benchmark functions
- 23 classical benchmark functions
- Custom objective functions

### 3. Built-in Test Functions
- Rosenbrock function
- Alpine function
- Ackley's function
- Easom function
- Eggcrate function
- Four-peak function

## Usage

### 1. Running Benchmark Tests

#### CEC Benchmark Tests
```bash
python master.py
```
This will run tests on CEC 2017 and 2019 benchmark functions using parallel processing.

#### Classical Benchmark Tests
```bash
python master_classical.py
```
This will run tests on classical benchmark functions with various dimensions.

### 2. Using Custom Objective Functions

To use the optimizer with custom objective functions:

1. Define your objective function:
```python
def custom_function(x):
    # Your objective function implementation
    return result
```

2. Define bounds:
```python
LB = np.array([lower_bound1, lower_bound2, ...])
UB = np.array([upper_bound1, upper_bound2, ...])
```

3. Run the optimizer:
```python
Best_FF, Best_P, conv = AO(population_size, M_Iter, LB, UB, Dim, custom_function)
```

### 3. Key Parameters

- `population_size`: Number of solutions in the population (default: 100)
- `M_Iter`: Maximum number of iterations (default: 100)
- `Dim`: Problem dimensions
- `LB`: Lower bounds array
- `UB`: Upper bounds array
- `F_obj`: Objective function to optimize

## Results

Test results are saved in the `outputs/` directory in JSON format, containing:
- Mean solution
- Standard deviation
- Best solution
- Worst solution
- Convergence history

## Dependencies

- numpy
- mpire (for parallel processing)
- opfunu (for benchmark functions)
- mealpy (for optimization utilities)

## Acknowledgments

We would like to thank the authors of the original Aquila Optimizer paper for their groundbreaking work in meta-heuristic optimization algorithms. This implementation is a reproducibility study of their work.

## Contributing

Feel free to submit issues and enhancement requests!

## License

This project is open source and available under the MIT License. See the [LICENSE](LICENSE) file for details.


