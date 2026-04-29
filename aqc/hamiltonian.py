from typing import Dict, Tuple
import numpy as np
from qiskit.quantum_info import SparsePauliOp

def qubo_to_ising(Q: np.ndarray) -> Tuple[Dict, np.ndarray, float]:
    """
    Convert a QUBO matrix Q to Ising model parameters.

    Args:
        Q: n x n symmetric matrix representing the QUBO problem.
    Returns:
        J: Dict mapping (i,j) to coupling strength for Z_i Z_j terms
        h: Field vector
        C: constant coefficient
    """
    n = Q.shape[0]

    zz_coeffs = {}
    for i in range(n):
        for j in range(i + 1, n):
            coeff = (Q[i, j] + Q[j, i]) / 4.0
            if abs(coeff) > 1e-12:
                zz_coeffs[(i, j)] = coeff

    z_coeffs = np.zeros(n)
    for i in range(n):
        z_coeffs[i] = -Q[i, i] / 2.0
        for j in range(n):
            if i != j:
                z_coeffs[i] -= (Q[i, j] + Q[j, i]) / 4.0

    C = sum(Q[i, i] / 2.0 for i in range(n))
    for i in range(n):
        for j in range(n):
            if i != j:
                C += Q[i, j] / 4.0

    return zz_coeffs, z_coeffs, C

def build_problem_hamiltonian(zz_coeffs: dict, z_coeffs: np.ndarray, n: int) -> SparsePauliOp:
    """
    Build the problem Hamiltonian as a SparsePauliOp:

    Args:
        zz_coeffs: Coupling strength for Z_i Z_j terms
        z_coeffs: Field vector for Z_i terms
        n: number of qubits
    """
    pauli_terms = []

    for (i, j), coeff in zz_coeffs.items():
        pauli_terms.append(("ZZ", [i, j], coeff))

    for i in range(n):
        if abs(z_coeffs[i]) > 1e-12:
            pauli_terms.append(("Z", [i], z_coeffs[i]))

    if not pauli_terms:
        pauli_terms.append(("I" * n, list(range(n)), 0.0))

    return SparsePauliOp.from_sparse_list(pauli_terms, num_qubits=n).simplify()

def build_driver_hamiltonian(n: int) -> SparsePauliOp:
    """
    Build the initial hamiltonian with a trivial ground state. 

    Args:
        n: number of qubits

    Returns:
        SparsePauliOp representing H_driver = -sum X_i
    """
    pauli_terms = [("X", [i], -1.0) for i in range(n)]
    return SparsePauliOp.from_sparse_list(pauli_terms, num_qubits=n).simplify()
