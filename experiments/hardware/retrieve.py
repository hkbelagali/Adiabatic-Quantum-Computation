import sys
import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from aqc.containers import Query
from aqc.utils import generate_random_qubo, brute_force_ground_state, success_probability
from aqc.aqc import build_aqc, simulate_aqc

HERE = Path(__file__).parent
CACHE_FILE = HERE / "hw_cache.json"


def counts_from_sampler_result(pub_result):
    raw = pub_result.data.meas.get_counts()
    return {bs[::-1]: cnt for bs, cnt in raw.items()}

def fetch_from_ibm(meta):
    import mthree
    from qiskit_ibm_runtime import QiskitRuntimeService

    backend_name = meta["backend"]
    N = meta["n"]
    instances = meta["instances"]
    trotter_steps = meta["trotter_steps"]
    job_ids = meta["job_ids"]

    service = QiskitRuntimeService()
    backend = service.backend(backend_name)
    mit = mthree.M3Mitigation(backend)
    mit.cals_from_system(range(N))

    cache = {}
    for inst in instances:
        job = service.job(job_ids[f"inst_{inst}"])
        job_result = job.result()

        hw_counts_list = []
        mit_probs_list = []

        for ci in range(len(trotter_steps)):
            hw_counts = counts_from_sampler_result(job_result[ci])
            hw_counts_list.append(hw_counts)

            be_counts = {bs[::-1]: cnt for bs, cnt in hw_counts.items()}
            quasi = mit.apply_correction(be_counts, range(N))
            mit_probs = {bs[::-1]: max(0.0, v) for bs, v in quasi.nearest_probability_distribution().items()}
            mit_probs_list.append(mit_probs)

        cache[str(inst)] = {"hw_counts": hw_counts_list, "mit_probs": mit_probs_list}

    with open(CACHE_FILE, "w") as f:
        json.dump(cache, f, default=lambda x: float(x) if isinstance(x, np.floating) else x)
    return cache

def aer_success(Q, steps, T, shots, ground_bs):
    query = Query(Q=Q, iterations=steps, total_time=T, shots=shots, trotter_order=2)
    circuit, eigseries = build_aqc(query)
    result = simulate_aqc(circuit, query, eigseries)
    return success_probability(result.counts, ground_bs, shots)

def main():
    with open(HERE / "job_ids.json") as f:
        meta = json.load(f)

    N = meta["n"]
    instances = meta["instances"]
    trotter_steps = meta["trotter_steps"]
    T = meta["T"]
    shots = meta["shots"]

    if CACHE_FILE.exists():
        try:
            with open(CACHE_FILE) as f:
                cache = json.load(f)
        except json.JSONDecodeError:
            print("Cache file corrupted, re-fetching from IBM...")
            cache = fetch_from_ibm(meta)
    else:
        cache = fetch_from_ibm(meta)

    results = {}
    for inst in instances:
        seed = N * 100 + inst
        Q = generate_random_qubo(N, seed=seed)
        _, ground_vec = brute_force_ground_state(Q)
        ground_bs = "".join(str(b) for b in ground_vec)

        inst_cache = cache[str(inst)]
        raw_success = []
        mit_success = []
        sim_success = []

        for ci, steps in enumerate(trotter_steps):
            hw_counts = inst_cache["hw_counts"][ci]
            total = sum(hw_counts.values())
            raw_success.append(success_probability(hw_counts, ground_bs, total))

            mit_probs = inst_cache["mit_probs"][ci]
            mit_success.append(mit_probs.get(ground_bs, 0.0))

            sim_success.append(aer_success(Q, steps, T, shots, ground_bs))

        results[inst] = dict(raw=raw_success, mitigated=mit_success, simulator=sim_success)

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
        ax.set_xlabel("Trotter steps")
        ax.set_ylabel("Success probability")
        ax.set_ylim(0, 1.05)
        ax.legend(loc="best")

    fig.suptitle("Hardware vs Simulator Success Probability")
    plt.tight_layout()
    plt.savefig(HERE / "hardware_results.pdf", dpi=200)
    plt.close()

    if circuit_depths:
        fig2, axes2 = plt.subplots(1, len(instances), figsize=(8 * len(instances), 9))
        if len(instances) == 1:
            axes2 = [axes2]

        for ax, inst in zip(axes2, instances):
            r = results[inst]
            depths = circuit_depths.get(f"inst_{inst}")
            if depths is None:
                continue
            ax.plot(depths, r["simulator"], marker="o", linewidth=2, label="Aer (noiseless)")
            ax.plot(depths, r["raw"],       marker="s", linewidth=2, linestyle="--", label="Hardware raw")
            ax.plot(depths, r["mitigated"], marker="^", linewidth=2, linestyle=":",  label="Hardware mitigated")
            ax.set_xlabel("Transpiled circuit depth")
            ax.set_ylabel("Success probability")
            ax.set_ylim(0, 1.05)
            ax.legend()

        fig2.suptitle("Success Probability vs Circuit Depth")
        plt.tight_layout()
        plt.savefig(HERE / "hardware_depth.pdf", dpi=200)
        plt.close()


if __name__ == "__main__":
    main()
