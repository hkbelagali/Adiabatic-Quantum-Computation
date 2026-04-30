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
N_INSTANCES = 100
TROTTER_STEPS = [10, 25, 50, 100, 200]
TROTTER_ORDERS = [1, 2]
T = 10.0
SHOTS = 4096

def run_size(n: int):
    n_steps  = len(TROTTER_STEPS)
    n_orders = len(TROTTER_ORDERS)
    success_probs = np.zeros((N_INSTANCES, n_steps, n_orders))

    for inst in range(N_INSTANCES):
        seed = n * 100 + inst
        Q = generate_random_qubo(n, seed=seed)
        _, ground_vec = brute_force_ground_state(Q)
        ground_bs = "".join(str(b) for b in ground_vec)

        for si, steps in enumerate(TROTTER_STEPS):
            for oi, order in enumerate(TROTTER_ORDERS):
                query = Query(Q=Q, iterations=steps, total_time=T,
                              shots=SHOTS, trotter_order=order)
                circuit, eigseries = build_aqc(query)
                result = simulate_aqc(circuit, query, eigseries)
                success_probs[inst, si, oi] = success_probability(
                    result.counts, ground_bs, SHOTS
                )

    np.savez(
        HERE / f"results_n{n}.npz",
        steps=np.array(TROTTER_STEPS),
        orders=np.array(TROTTER_ORDERS),
        success_probs=success_probs,
    )
    return success_probs

def _sa_mean_by_n():
    sa_data  = np.load(HERE.parent / "sa_sweep" / "results.npz", allow_pickle=True)
    labels   = sa_data["labels"].astype(str)
    n_labels = sa_data["n_labels"]
    rates    = sa_data["success_rates"]
    random_mask = np.array([lbl.startswith("random") for lbl in labels])
    return {
        n: float(rates[random_mask & (n_labels == n)].mean())
        for n in N_SIZES
    }

def plot_all():
    plt.style.use(str(ROOT / "presentation.mplstyle"))
    fig, axes = plt.subplots(2, 2, figsize=(18, 14))
    order_labels = {1: "1st order (Lie-Trotter)", 2: "2nd order (Suzuki-Trotter)"}

    sa_means = _sa_mean_by_n()

    for ax, n in zip(axes.flat, N_SIZES):
        data = np.load(HERE / f"results_n{n}.npz")
        steps = data["steps"]
        sp    = data["success_probs"]

        for oi, order in enumerate(TROTTER_ORDERS):
            mean = sp[:, :, oi].mean(axis=0)
            std  = sp[:, :, oi].std(axis=0)
            ax.errorbar(steps, mean, yerr=std, marker="o", linewidth=2,
                        capsize=4, label=order_labels[order])

        ax.axhline(sa_means[n], color="C3", linestyle="--", linewidth=1.5,
                   label=f"SA mean ({sa_means[n]:.2f})")
        ax.set_title(f"n = {n}")
        ax.set_xlabel("Trotter steps")
        ax.set_xlim(0, max(TROTTER_STEPS) * 1.1)
        ax.set_ylabel("Success probability")
        ax.set_ylim(0, 1.05)
        ax.legend(loc='best')

    fig.suptitle(f"Success Probability vs Trotter Steps")
    plt.tight_layout()
    out = HERE / "trotter_steps.pdf"
    plt.savefig(out, dpi=200)
    plt.close()

if __name__ == "__main__":
    # for n in N_SIZES:
        # run_size(n)
    plot_all()
