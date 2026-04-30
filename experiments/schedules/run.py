import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from aqc.containers import Query
from aqc.utils import generate_random_qubo, brute_force_ground_state, success_probability
from aqc.aqc import build_aqc, simulate_aqc

HERE = Path(__file__).parent

N_SIZES = [3, 4, 5, 6]
N_INSTANCES = 10
T_VALUES = [0.1, 0.5, 1.0, 5.0, 10.0, 50.0, 100.0]
SCHEDULES = ["linear", "quadratic", "sinusoidal"]
ITERATIONS = 50
ORDER = 2
SHOTS = 4096

def run_size(n: int):
    success_probs = np.zeros((N_INSTANCES, len(T_VALUES), len(SCHEDULES)))

    for inst in range(N_INSTANCES):
        seed = n * 100 + inst
        Q = generate_random_qubo(n, seed=seed)
        _, ground_vec = brute_force_ground_state(Q)
        ground_bs = "".join(str(b) for b in ground_vec)

        for ti, T in enumerate(T_VALUES):
            for si, sched in enumerate(SCHEDULES):
                query = Query(Q=Q, iterations=ITERATIONS, total_time=T,
                              shots=SHOTS, trotter_order=ORDER, schedule=sched)
                circuit, eigseries = build_aqc(query)
                result = simulate_aqc(circuit, query, eigseries)
                success_probs[inst, ti, si] = success_probability(
                    result.counts, ground_bs, SHOTS
                )

    np.savez(
        HERE / f"results_n{n}.npz",
        T_values=np.array(T_VALUES),
        schedules=np.array(SCHEDULES),
        success_probs=success_probs,
    )
    return success_probs

def plot_all():
    plt.style.use(str(ROOT / "presentation.mplstyle"))
    fig, axes = plt.subplots(2, 2)
    markers = ["o", "s", "^"]

    for ax, n in zip(axes.flat, N_SIZES):
        data   = np.load(HERE / f"results_n{n}.npz")
        T_vals = data["T_values"]
        sp     = data["success_probs"]

        for si, sched in enumerate(SCHEDULES):
            mean = sp[:, :, si].mean(axis=0)
            std  = sp[:, :, si].std(axis=0)
            ax.errorbar(T_vals, mean, yerr=std, marker=markers[si],
                        linewidth=2, capsize=4, label=sched)

        ax.set_xscale("log")
        ax.set_title(f"n = {n}")
        ax.set_xlabel("Evolution time T")
        ax.set_ylabel("Success probability")
        ax.set_ylim(0, 1.05)
        ax.legend(loc='best', numpoints=1)

    fig.suptitle(
        f"Schedule Comparison (iterations={ITERATIONS}, order={ORDER}, shots={SHOTS})"
    )
    plt.tight_layout()
    plt.savefig(HERE / "schedules.pdf", dpi=200)
    plt.close()

if __name__ == "__main__":
    for n in N_SIZES:
        run_size(n)
    plot_all()
