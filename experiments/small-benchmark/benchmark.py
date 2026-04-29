import matplotlib.pyplot as plt
import numpy as np 
from pathlib import Path

from aqc.containers import Query
from aqc.utils import brute_force_ground_state, adiabatic_ground_state_energy
from aqc.aqc import build_aqc, simulate_aqc

def save_energy_plot(num_qubits, brute_force_means, adiabatic_means, output_base: Path):
    output_path = output_base.with_suffix(".pdf")
    plt.plot(num_qubits, brute_force_means, marker="o", markersize=10, linewidth=3, label="Brute-force minimum cost")
    plt.plot(num_qubits, adiabatic_means, marker="s", markersize=8, linewidth=3, linestyle='--',label="AQC minimum cost")
    plt.xlabel("QUBO size")
    plt.ylabel("Average cost")
    plt.title("Average Cost vs QUBO Size (N=10 trials)")
    plt.grid(True, alpha=0.3)
    plt.minorticks_off()
    plt.xlim(1.5, 6.5)
    plt.ylim(np.min([brute_force_means.min(), adiabatic_means.min()]) - 5, np.max([brute_force_means.max(), adiabatic_means.max()]) + 5)
    plt.legend(numpoints=1)
    plt.tight_layout()
    plt.savefig(output_path, dpi=800)
    plt.close()
    return output_path

if __name__ == "__main__":

    num_qubits = list(range(2, 7))
    N_TRIALS = 10

    brute_force_means = np.zeros(len(num_qubits), dtype=float)
    adiabatic_means = np.zeros(len(num_qubits), dtype=float)

    for qubit_index, N in enumerate(num_qubits):
        brute_force_trials = []
        adiabatic_trials = []

        for trial in range(N_TRIALS):
            Q = np.random.uniform(-10, 10, (N, N))
            Q = (Q + Q.T) / 2  # make symmetric

            query = Query(
                Q=Q,
                iterations=200,
                total_time=20.0,
                shots=2048,
                trotter_order=2,
                trotter_reps=1
            )

            brute_force_energy, _ = brute_force_ground_state(Q)

            exec, eigseries = build_aqc(query)
            result = simulate_aqc(exec, query, eigseries)
            adiabatic_energy, _ = adiabatic_ground_state_energy(Q, result.counts)

            brute_force_trials.append(brute_force_energy)
            adiabatic_trials.append(adiabatic_energy)

            print(
                f"N={N}, trial={trial + 1}/{N_TRIALS}: "
                f"brute-force={brute_force_energy:.6f}, "
                f"adiabatic={adiabatic_energy:.6f}"
            )

        brute_force_means[qubit_index] = float(np.mean(brute_force_trials))
        adiabatic_means[qubit_index] = float(np.mean(adiabatic_trials))

        print(
            f"Average over N={N}: "
            f"brute-force={brute_force_means[qubit_index]:.6f}, "
            f"adiabatic={adiabatic_means[qubit_index]:.6f}"
        )

    np.savez("energy_comparison.npz", num_qubits=num_qubits, brute_force_means=brute_force_means, adiabatic_means=adiabatic_means)

    plt.style.use("../presentation.mplstyle")
    data = np.load("energy_comparison.npz")
    num_qubits = data["num_qubits"]
    brute_force_means = data["brute_force_means"]
    adiabatic_means = data["adiabatic_means"]

    figure_path = save_energy_plot(num_qubits, brute_force_means, adiabatic_means, Path("energy_comparison"))