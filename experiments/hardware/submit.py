import sys
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2 as Sampler
from qiskit_ibm_runtime.options import SamplerOptions

from aqc.containers import Query
from aqc.utils import generate_random_qubo, brute_force_ground_state, success_probability
from aqc.aqc import build_aqc, simulate_aqc

HERE     = Path(__file__).parent
EXP2_DIR = HERE.parent / "evolution_time"

N            = 3
N_INSTANCES  = 10
TROTTER_STEPS = [5, 10, 20, 40, 80]
TOP_K        = 3        
SHOTS        = 4096

def pick_top_instances(k: int):
    data = np.load(EXP2_DIR / f"results_n{N}.npz")
    sp   = data["success_probs"]
    T_vals = data["T_values"]
    T_idx = int(np.argmin(np.abs(T_vals - 10.0)))
    scores = sp[:, T_idx]
    top_idx = np.argsort(scores)[::-1][:k]
    return [int(i) for i in top_idx], 10.0

def main():
    service = QiskitRuntimeService()
    backend = service.least_busy(simulator=False, operational=True, min_num_qubits=N)
    pm = generate_preset_pass_manager(optimization_level=1, backend=backend)

    top_instances, T = pick_top_instances(TOP_K)

    job_ids       = {}
    circuit_depths = {}

    for inst in top_instances:
        seed = N * 100 + inst
        Q    = generate_random_qubo(N, seed=seed)

        circuits = []
        depths   = []
        for steps in TROTTER_STEPS:
            query   = Query(Q=Q, iterations=steps, total_time=T,
                            shots=SHOTS, trotter_order=2)
            circuit, _ = build_aqc(query)
            circuit.measure_all()
            transpiled = pm.run(circuit)
            depths.append(transpiled.depth())
            circuits.append(transpiled)

        circuit_depths[f"inst_{inst}"] = depths

        options = SamplerOptions()
        options.default_shots = SHOTS

        sampler = Sampler(backend, options=options)
        job     = sampler.run(circuits)
        job_ids[f"inst_{inst}"] = job.job_id()

    meta = {
        "backend": backend.name,
        "n": N,
        "instances": top_instances,
        "trotter_steps": TROTTER_STEPS,
        "T": T,
        "shots": SHOTS,
        "job_ids": job_ids,
        "circuit_depths": circuit_depths,
    }
    with open(HERE / "job_ids.json", "w") as f:
        json.dump(meta, f, indent=2)

if __name__ == "__main__":
    main()
