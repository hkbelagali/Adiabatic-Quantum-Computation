import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from aqc.utils import generate_random_qubo, brute_force_ground_state, compute_spectral_gap

HERE = Path(__file__).parent
EXP2_DIR = HERE.parent / "evolution_time"

N_SIZES = [3, 4, 5, 6]
N_INSTANCES = 100
N_POINTS = 100
T_VALUES = [0.1, 0.5, 1.0, 5.0, 10.0, 50.0, 100.0]
SUCCESS_THRESHOLD = 0.80


def interpolate_T_required(T_vals: np.ndarray, sp_instance: np.ndarray) -> float:
    for i in range(len(T_vals) - 1):
        if sp_instance[i] < SUCCESS_THRESHOLD <= sp_instance[i + 1]:

            t0, t1 = T_vals[i], T_vals[i + 1]
            p0, p1 = sp_instance[i], sp_instance[i + 1]
            return float(t0 + (t1 - t0) * (SUCCESS_THRESHOLD - p0) / (p1 - p0))
    if sp_instance[-1] >= SUCCESS_THRESHOLD:
        return float(T_vals[-1])
    return float("nan")


def run():
    all_min_gaps = []
    all_s_at_min = []
    all_success_p = []
    all_T_required = []
    all_gap_product = []
    all_n = []

    for n in N_SIZES:
        exp2_data = np.load(EXP2_DIR / f"results_n{n}.npz")
        T_vals = exp2_data["T_values"]
        sp_matrix = exp2_data["success_probs"]

        for inst in range(N_INSTANCES):
            seed = n * 100 + inst
            Q = generate_random_qubo(n, seed=seed)

            ts, gaps = compute_spectral_gap(Q, n_points=N_POINTS)
            min_idx = int(np.argmin(gaps))
            min_gap = float(gaps[min_idx])
            s_at_min = float(ts[min_idx])

            sp_at_maxT = float(sp_matrix[inst, 4])  

            T_req = interpolate_T_required(T_vals, sp_matrix[inst])
            gap_product = min_gap ** 2 * T_req if not np.isnan(T_req) else float("nan")

            all_min_gaps.append(min_gap)
            all_s_at_min.append(s_at_min)
            all_success_p.append(sp_at_maxT)
            all_T_required.append(T_req)
            all_gap_product.append(gap_product)
            all_n.append(n)

    results = dict(
        min_gaps=np.array(all_min_gaps),
        s_at_min_gaps=np.array(all_s_at_min),
        success_probs=np.array(all_success_p),
        T_required=np.array(all_T_required),
        gap_T_product=np.array(all_gap_product),
        n_labels=np.array(all_n),
    )
    np.savez(HERE / "results.npz", **results)
    return results


def plot(results: dict):
    plt.style.use(str(ROOT / "presentation.mplstyle"))
    min_gaps = results["min_gaps"]
    s_at_min = results["s_at_min_gaps"]
    success_p = results["success_probs"]
    gap_product = results["gap_T_product"]
    n_labels = results["n_labels"]

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(30, 9))

    for n in N_SIZES:
        mask = n_labels == n
        ax1.scatter(min_gaps[mask], success_p[mask], label=f"n={n}", s=80, color=f"C{n-3}")
    ax1.set_xlabel(r"Minimum spectral gap $\Delta$")
    ax1.set_ylabel("Success probability")
    ax1.set_title(r"Success probability vs $\Delta$")
    ax1.legend(scatterpoints=1, loc='best')

    valid = ~np.isnan(gap_product)
    for n in N_SIZES:
        mask = (n_labels == n) & valid
        ax2.scatter(min_gaps[mask], gap_product[mask], label=f"n={n}", s=80, color=f"C{n-3}")
    ax2.set_xlabel(r"Minimum spectral gap $\Delta$")
    ax2.set_ylabel(r"$\Delta^2 \times T_{\text{required}}$")
    ax2.set_title(r"Adiabatic condition product")
    ax2.legend(scatterpoints=1, loc='best')

    bins = np.linspace(0, 1, 21)
    for n in N_SIZES:
        mask = n_labels == n
        ax3.hist(s_at_min[mask], bins=bins, alpha=0.5, label=f"n={n}", color=f"C{n-3}")
    ax3.axvline(0.5, color="k", linestyle="--", linewidth=1.5, label="s = 0.5")
    ax3.set_xlabel(r"Location of minimum gap $s^*$")
    ax3.set_ylabel("Count")
    ax3.set_title(r"Distribution of $s^*$")
    ax3.legend(loc='best')

    plt.tight_layout()
    out = HERE / "spectral_gap.pdf"
    plt.savefig(out, dpi=200)
    plt.close()

if __name__ == "__main__":
    # results = run()
    results = np.load(HERE / "results.npz")
    plot(results)
