import numpy as np
from typing import Tuple

def qubo_energy(Q: np.ndarray, x: np.ndarray) -> float:
    return float(x.T @ Q @ x)


def bitstring_to_vector(bitstring: str) -> np.ndarray:
    cleaned = bitstring.replace(" ", "")
    return np.array([int(bit) for bit in cleaned], dtype=int)

def brute_force_ground_state(Q: np.ndarray) -> Tuple[float, np.ndarray]:
    n = Q.shape[0]
    best_energy = np.inf
    best_bitstring = np.zeros(n, dtype=int)

    for i in range(2**n):
        x = np.array([int(bit) for bit in bin(i)[2:].zfill(n)], dtype=int)
        energy = qubo_energy(Q, x)
        if energy < best_energy:
            best_energy = energy
            best_bitstring = x

    return best_energy, best_bitstring

def adiabatic_ground_state_energy(Q: np.ndarray, counts: dict):
    best_energy = np.inf
    best_bitstring = None

    for bitstring, count in counts.items():
        if count <= 0:
            continue

        x = bitstring_to_vector(bitstring)
        energy = qubo_energy(Q, x)
        if energy < best_energy:
            best_energy = energy
            best_bitstring = x

    if best_bitstring is None:
        raise ValueError("No measured bitstrings were returned by the annealing run.")

    return best_energy, best_bitstring


def generate_random_qubo(n: int, seed=None, low: float = -10.0, high: float = 10.0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    Q = rng.uniform(low, high, (n, n))
    return (Q + Q.T) / 2


def generate_maxcut_qubo(G) -> np.ndarray:
    """Return QUBO matrix for minimizing negative Max-Cut on graph G."""
    n = G.number_of_nodes()
    Q = np.zeros((n, n))
    for i, j in G.edges():
        Q[i, i] -= 1.0
        Q[j, j] -= 1.0
        Q[i, j] += 1.0
        Q[j, i] += 1.0
    return Q


def compute_spectral_gap(Q: np.ndarray, n_points: int = 100) -> Tuple[np.ndarray, np.ndarray]:
    """Compute the instantaneous spectral gap of H_adiabatic(s) at n_points values of s in [0,1]."""
    from aqc.hamiltonian import qubo_to_ising, build_problem_hamiltonian, build_driver_hamiltonian
    n = Q.shape[0]
    J, h, _ = qubo_to_ising(Q)
    H_d = build_driver_hamiltonian(n)
    H_p = build_problem_hamiltonian(J, h, n)
    ts = np.linspace(0, 1, n_points)
    gaps = np.empty(n_points)
    for idx, s in enumerate(ts):
        H = ((1 - s) * H_d + s * H_p).simplify()
        eigs = np.sort(np.real(np.linalg.eigvals(H.to_matrix())))
        gaps[idx] = eigs[1] - eigs[0]
    return ts, gaps


def success_probability(counts: dict, ground_bitstring: str, total_shots: int) -> float:
    """Fraction of shots that landed on the ground state bitstring."""
    return counts.get(ground_bitstring, 0) / total_shots

