import sys
import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import mthree
from qiskit_ibm_runtime import QiskitRuntimeService

from aqc.containers import Query
from aqc.utils import generate_random_qubo, brute_force_ground_state, success_probability
from aqc.aqc import build_aqc, simulate_aqc
from qiskit_aer import Aer

HERE = Path(__file__).parent


def counts_from_sampler_result(pub_result, n_qubits: int):
    raw = pub_result.data.meas.get_counts()
    return {bs[::-1]: cnt for bs, cnt in raw.items()}


def aer_success(Q, steps, T, shots, ground_bs):
    query   = Query(Q=Q, iterations=steps, total_time=T, shots=shots, trotter_order=2)
    circuit, eigseries = build_aqc(query)
    result  = simulate_aqc(circuit, query, eigseries)
    return success_probability(result.counts, ground_bs, shots)


def main():
    with open(HERE / "job_ids.json") as f:
        meta = json.load(f)

    backend_name  = meta["backend"]
    N             = meta["n"]
    instances     = meta["instances"]
    trotter_steps = meta["trotter_steps"]
    T             = meta["T"]
    shots         = meta["shots"]
    job_ids       = meta["job_ids"]

    service = QiskitRuntimeService()
    backend = service.backend(backend_name)
    mit = mthree.M3Mitigation(backend)
    mit.cals_from_system(range(N))

    results = {}

    for inst in instances:
        seed = N * 100 + inst
        Q    = generate_random_qubo(N, seed=seed)
        _, ground_vec = brute_force_ground_state(Q)
        ground_bs     = "".join(str(b) for b in ground_vec)

        job = service.job(job_ids[f"inst_{inst}"])
        job_result = job.result()

        raw_success = []
        mit_success = []
        sim_success = []

        for ci, steps in enumerate(trotter_steps):
            pub_result = job_result[ci]
            hw_counts  = counts_from_sampler_result(pub_result, N)

            total = sum(hw_counts.values())
            raw_p = success_probability(hw_counts, ground_bs, total)

            be_counts = {bs[::-1]: cnt for bs, cnt in hw_counts.items()}
            quasi     = mit.apply_correction(be_counts, range(N))
            mit_counts = {bs[::-1]: max(0, v) for bs, v in quasi.nearest_probability_distribution().items()}
            mit_p      = mit_counts.get(ground_bs, 0.0)

            sim_p = aer_success(Q, steps, T, shots, ground_bs)

            raw_success.append(raw_p)
            mit_success.append(mit_p)
            sim_success.append(sim_p)

        results[inst] = dict(
            raw=raw_success, mitigated=mit_success, simulator=sim_success
        )

    circuit_depths = meta.get("circuit_depths", {})

    save_dict = {"trotter_steps": trotter_steps, "T": T, "instances": instances}
    for inst in instances:
        for key in ("raw", "mitigated", "simulator"):
            save_dict[f"inst{inst}_{key}"] = results[inst][key]
        if f"inst_{inst}" in circuit_depths:
            save_dict[f"inst{inst}_depths"] = np.array(circuit_depths[f"inst_{inst}"])
    np.savez(HERE / "results.npz", **save_dict)

    plt.style.use(str(ROOT / "presentation.mplstyle"))
    fig, axes = plt.subplots(1, len(instances), figsize=(8 * len(instances), 9))
    if len(instances) == 1:
        axes = [axes]

    for ax, inst in zip(axes, instances):
        r = results[inst]
        ax.plot(trotter_steps, r["simulator"], marker="o", linewidth=2, label="Aer (noiseless)")
        ax.plot(trotter_steps, r["raw"],       marker="s", linewidth=2, linestyle="--", label="Hardware raw")
        ax.plot(trotter_steps, r["mitigated"], marker="^", linewidth=2, linestyle=":",  label="Hardware mitigated")
        ax.set_title(f"Instance {inst}")
        ax.set_xlabel("Trotter steps")
        ax.set_ylabel("Success probability")
        ax.set_ylim(0, 1.05)
        ax.legend(loc='best')

    fig.suptitle(f"Hardware vs Simulator (n={N}, T={T}, backend={backend_name})")
    plt.tight_layout()
    plt.savefig(HERE / "hardware_results.pdf", dpi=200)
    plt.close()

    if circuit_depths:
        fig2, axes2 = plt.subplots(1, len(instances), figsize=(8 * len(instances), 9))
        if len(instances) == 1:
            axes2 = [axes2]

        for ax, inst in zip(axes2, instances):
            r      = results[inst]
            depths = circuit_depths.get(f"inst_{inst}")
            if depths is None:
                continue
            ax.plot(depths, r["simulator"], marker="o", linewidth=2, label="Aer (noiseless)")
            ax.plot(depths, r["raw"],       marker="s", linewidth=2, linestyle="--", label="Hardware raw")
            ax.plot(depths, r["mitigated"], marker="^", linewidth=2, linestyle=":",  label="Hardware mitigated")
            ax.set_title(f"Instance {inst}")
            ax.set_xlabel("Transpiled circuit depth")
            ax.set_ylabel("Success probability")
            ax.set_ylim(0, 1.05)
            ax.legend()

        fig2.suptitle(f"Success Probability vs Circuit Depth (n={N}, T={T}, backend={backend_name})")
        plt.tight_layout()
        plt.savefig(HERE / "hardware_depth.pdf", dpi=200)
        plt.close()

if __name__ == "__main__":
    main()
