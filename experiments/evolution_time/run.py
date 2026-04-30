"""
Fixed Trotter steps and varying total evolution time T
"""
import sys
import os
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from aqc.containers import Query
from aqc.utils import generate_random_qubo, brute_force_ground_state, success_probability
from aqc.aqc import build_aqc, simulate_aqc

HERE = Path(__file__).parent

N_SIZES     = [3, 4, 5, 6]
N_INSTANCES = 100
T_VALUES    = [0.1, 0.5, 1.0, 5.0, 10.0, 50.0, 100.0]
ITERATIONS  = 50
ORDER       = 2
SHOTS       = 4096


def run_instance(args):
    n, inst = args
    seed = n * 100 + inst
    Q = generate_random_qubo(n, seed=seed)
    _, ground_vec = brute_force_ground_state(Q)
    ground_bs = "".join(str(b) for b in ground_vec)

    row = np.zeros(len(T_VALUES))
    for ti, T in enumerate(T_VALUES):
        query = Query(Q=Q, iterations=ITERATIONS, total_time=T,
                      shots=SHOTS, trotter_order=ORDER)
        circuit, eigseries = build_aqc(query)
        result = simulate_aqc(circuit, query, eigseries)
        row[ti] = success_probability(result.counts, ground_bs, SHOTS)

    return n, inst, row


def run_all():
    all_tasks = [(n, inst) for n in N_SIZES for inst in range(N_INSTANCES)]
    results = {n: np.zeros((N_INSTANCES, len(T_VALUES))) for n in N_SIZES}

    n_workers = os.cpu_count() or 4
    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        futures = {executor.submit(run_instance, t): t for t in all_tasks}
        for future in as_completed(futures):
            n, inst, row = future.result()
            results[n][inst] = row

    for n in N_SIZES:
        np.savez(
            HERE / f"results_n{n}.npz",
            T_values=np.array(T_VALUES),
            success_probs=results[n],
        )

    return results


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

    sa_means = _sa_mean_by_n()

    for ax, n in zip(axes.flat, N_SIZES):
        data = np.load(HERE / f"results_n{n}.npz")
        T_vals = data["T_values"]
        sp     = data["success_probs"]
        mean   = sp.mean(axis=0)
        std    = sp.std(axis=0)

        ax.errorbar(T_vals, mean, yerr=std, marker="o", linewidth=2, capsize=4, label="AQC mean ± std")
        ax.axhline(sa_means[n], color="C3", linestyle="--", linewidth=1.5, label=f"SA mean ({sa_means[n]:.2f})")
        ax.set_xscale("log")
        ax.set_title(f"n = {n}")
        ax.set_xlabel("Evolution time T")
        ax.set_ylabel("Success probability")
        ax.set_ylim(0, 1.05)
        ax.legend(loc="best")

    fig.suptitle(
        f"Success Probability vs Evolution Time "
        f"(iterations={ITERATIONS}, order={ORDER}, shots={SHOTS})"
    )
    plt.tight_layout()
    plt.savefig(HERE / "evolution_time.pdf", dpi=200)
    plt.close()


if __name__ == "__main__":
    run_all()
    plot_all()
