import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from aqc.utils import generate_random_qubo
from aqc.hamiltonian import qubo_to_ising, build_problem_hamiltonian, build_driver_hamiltonian

HERE = Path(__file__).parent

N = 6
SEED = 600 
N_POINTS = 200

SCHEDULES = {
    "linear":     lambda t: t,
    "quadratic":  lambda t: t ** 2,
    "sinusoidal": lambda t: float(np.sin(np.pi * t / 2) ** 2),
}

SCHEDULE_LABELS = {
    "linear":     r"Linear: $s(t) = \frac{t}{T}$",
    "quadratic":  r"Quadratic: $s(t) = \left (\frac{t}{T}\right )^2$",
    "sinusoidal": r"Sinusoidal: $s(t) = \sin^2\!\left(\tfrac{\pi t}{2T}\right)$",
}


def compute_eigenseries(Q: np.ndarray, schedule_fn, n_points: int = N_POINTS):
    n = Q.shape[0]
    J, h, _ = qubo_to_ising(Q)
    H_d = build_driver_hamiltonian(n)
    H_p = build_problem_hamiltonian(J, h, n)

    t_vals = np.linspace(0, 1, n_points)
    all_eigs = np.empty((n_points, 2 ** n))
    for i, t in enumerate(t_vals):
        s = schedule_fn(t)
        H = ((1 - s) * H_d + s * H_p).simplify()
        all_eigs[i] = np.sort(np.real(np.linalg.eigvals(H.to_matrix())))

    return t_vals, all_eigs


def plot():
    plt.style.use(str(ROOT / "presentation.mplstyle"))

    Q = generate_random_qubo(N, seed=SEED)

    fig, axes = plt.subplots(
        2, 3, figsize=(24, 15), sharex="col",
        gridspec_kw={"height_ratios": [3, 1]},
    )

    for col, (name, fn) in enumerate(SCHEDULES.items()):
        ax_eig = axes[0, col]
        ax_gap = axes[1, col]

        t_vals, eigs = compute_eigenseries(Q, fn)
        gap = eigs[:, 1] - eigs[:, 0]
        min_idx = int(np.argmin(gap))
        t_min = t_vals[min_idx]
        delta_min = gap[min_idx]

        for k in range(2, eigs.shape[1]):
            ax_eig.plot(t_vals, eigs[:, k], color="0.70", linewidth=0.8, alpha=0.5)

        ax_eig.plot(t_vals, eigs[:, 1], color="C1", linewidth=2.0, label=r"$E_1$")
        ax_eig.plot(t_vals, eigs[:, 0], color="C0", linewidth=2.5, label=r"$E_0$")
        ax_eig.axvline(t_min, color="C2", linestyle="--", linewidth=1.5,
                       label=rf"$t^* = {t_min:.2f}$")
        ax_eig.set_title(SCHEDULE_LABELS[name])
        ax_eig.set_ylabel("Energy")
        ax_eig.legend(loc="best")

        ax_gap.plot(t_vals, gap, color="k", linewidth=2.0)
        ax_gap.axvline(t_min, color="C2", linestyle="--", linewidth=1.5)
        ax_gap.scatter([t_min], [delta_min], color="C2", zorder=5, s=100,
                       label=rf"$\Delta_{{\min}} = {delta_min:.3f}$")
        ax_gap.set_xlabel(r"Time parameter $t$")
        ax_gap.set_ylabel(r"Spectral gap $\Delta$")
        ax_gap.legend(loc="best", scatterpoints=1)

    plt.tight_layout()
    out = HERE / "eigenseries.svg"
    plt.savefig(out, dpi=200)
    plt.close()


if __name__ == "__main__":
    plot()
