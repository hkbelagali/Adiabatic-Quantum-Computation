# Adiabatic Quantum Computation of QUBO Problems
In this project, I try to learn more about the behavior of Adiabatic Quantum Computation for solving Quadratic Unconstrained Binary Optimization problems. I systematically vary parameters that control algorithmic performance and test on the `Aer` noiseless simulator as well as IBM's Kingston quantum device to compare performance on simulators and real hardware.

![Adiabatic Evolution of Eigenvalues](experiments/eigenseries/eigenseries.png)
<p align="center">
Time series of the eigenvalues of $\mathcal{H}$ during adiabatic evolution.
</p>

# Usage 
The code for the AQC algorithm is in the `aqc/` directory. To install the minimal build, run 
```
pip install .
```
in the root directory of the project. There are also optional dependencies provided for development and running the experiments - 
```
pip install .[dev]
```
```
pip install .[experiments]
```
The details of the implementation are in the [report](report/Report.pdf).
