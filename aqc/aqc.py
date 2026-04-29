import numpy as np
from qiskit.circuit import QuantumCircuit, QuantumRegister
from qiskit.circuit.library import PauliEvolutionGate
from qiskit.synthesis import SuzukiTrotter, LieTrotter
from qiskit_aer import Aer
from pathlib import Path
import matplotlib.pyplot as plt

from aqc.hamiltonian import qubo_to_ising, build_problem_hamiltonian, build_driver_hamiltonian
from aqc.utils import qubo_energy, bitstring_to_vector
from aqc.containers import Query, Result

from typing import Tuple

def build_aqc(query: Query) -> Tuple[QuantumCircuit, np.ndarray]:
    """
    Build the adiabatic quantum computation circuit for a given QUBO problem.
    """
    Q = query.Q

    if Q.shape[0] != Q.shape[1]:
        raise ValueError("QUBO matrix must be square!")

    n = Q.shape[0]

    J, h, _ = qubo_to_ising(Q)

    H_driver = build_driver_hamiltonian(n)
    H_problem = build_problem_hamiltonian(J, h, n)

    qubits = QuantumRegister(n)
    circuit = QuantumCircuit(qubits)

    for i in range(n):
        circuit.h(i)

    iterations = query.iterations
    T = query.total_time
    timesteps = np.linspace(0, 1, iterations)
    delta_t = T / iterations  

    if query.trotter_order == 1:
        synthesis = LieTrotter(reps=query.trotter_reps)
    else:
        synthesis = SuzukiTrotter(
            order=query.trotter_order,
            reps=query.trotter_reps
        )

    eigseries = []

    for s in timesteps:
        H_adiabatic = (1 - s) * H_driver + s * H_problem
        H_adiabatic = H_adiabatic.simplify()

        H_matrix = H_adiabatic.to_matrix()
        eigs = np.sort(np.real(np.linalg.eigvals(H_matrix)))
        eigseries.append(eigs)


        evo_gate = PauliEvolutionGate(
            H_adiabatic,
            time=delta_t,
            synthesis=synthesis
        )
        circuit.append(evo_gate, qubits)

    return circuit, eigseries

def simulate_aqc(circuit: QuantumCircuit, query: Query, eigseries: np.ndarray) -> Result:
    sim = Aer.get_backend('aer_simulator')
    circuit_state = circuit.copy()
    circuit_state.save_statevector()

    state_result = sim.run(
        circuit_state.decompose(reps=3),
        shots=1
    ).result()

    circuit_counts = circuit.copy()
    circuit_counts.measure_all()
    counts_result = sim.run(
        circuit_counts.decompose(reps=3),
        shots=query.shots
    ).result()

    state = np.array(state_result.get_statevector())
    prob = np.abs(state) ** 2

    result_dict = {
        bin(i)[2:].zfill(n)[::-1]: prob[i] for i in range(prob.size)
    }
    eigseries = np.array(eigseries).T
    counts = {
        bitstring.replace(" ", "")[::-1]: count
        for bitstring, count in counts_result.get_counts().items()
    }

    return Result(
        result_dict,
        eigseries,
        counts,
        state
    )


