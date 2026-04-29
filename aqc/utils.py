import numpy as np 

def qubo_energy(Q: np.ndarray, x: np.ndarray) -> float:
    return float(x.T @ Q @ x)


def bitstring_to_vector(bitstring: str) -> np.ndarray:
    cleaned = bitstring.replace(" ", "")
    return np.array([int(bit) for bit in cleaned], dtype=int)

def brute_force_ground_state(Q: np.ndarray):
    n = Q.shape[0]
    best_energy = np.inf
    best_bitstring = None

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

