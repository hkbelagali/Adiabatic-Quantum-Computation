import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import networkx as nx

from aqc.utils import generate_random_qubo, generate_maxcut_qubo, brute_force_ground_state
from sa.annealer import Annealer

HERE = Path(__file__).parent

SA_ML = 100
SA_ITER = 500
SA_T0 = 10.0
SA_TF = 0.01
SA_TRIALS = 50

RANDOM_N_SIZES = [3, 4, 5, 6]
N_INSTANCES_RND = 10

STRUCT_N_SIZES = [4, 5, 6]
N_INSTANCES_STR = 10
GRAPH_TYPES = ["er_03", "er_05", "er_07", "3reg"]

def make_graph(gtype: str, n: int, seed: int):
    if gtype == "er_03":
        return nx.erdos_renyi_graph(n, 0.3, seed=seed)
    if gtype == "er_05":
        return nx.erdos_renyi_graph(n, 0.5, seed=seed)
    if gtype == "er_07":
        return nx.erdos_renyi_graph(n, 0.7, seed=seed)
    if gtype == "3reg":
        if (n * 3) % 2 != 0:
            return None
        return nx.random_regular_graph(3, n, seed=seed)
    raise ValueError(gtype)

def sa_run(Q: np.ndarray, ground_energy: float):
    ann = Annealer(ml=SA_ML, iterations=SA_ITER, T_0=SA_T0, T_f=SA_TF)
    hits = 0
    best_e = np.inf
    for _ in range(SA_TRIALS):
        ann.X = None
        sol = ann.anneal(Q)
        e = float(sol.x_min.T @ Q @ sol.x_min)
        if e < best_e:
            best_e = e
        if abs(e - ground_energy) < 1e-6:
            hits += 1
    return hits / SA_TRIALS, best_e

def run():
    records = []

    for n in RANDOM_N_SIZES:
        for inst in range(N_INSTANCES_RND):
            seed = n * 100 + inst
            Q = generate_random_qubo(n, seed=seed)
            ground_e, _ = brute_force_ground_state(Q)
            srate, best_e = sa_run(Q, ground_e)
            label = f"random_n{n}_i{inst}"
            records.append((label, n, srate, best_e, ground_e))

    for ni, n in enumerate(STRUCT_N_SIZES):
        for gi, gtype in enumerate(GRAPH_TYPES):
            for inst in range(N_INSTANCES_STR):
                seed = ni * 1000 + gi * 100 + inst
                G = make_graph(gtype, n, seed)
                if G is None or G.number_of_edges() == 0:
                    continue
                Q = generate_maxcut_qubo(G)
                ground_e, _ = brute_force_ground_state(Q)
                srate, best_e = sa_run(Q, ground_e)
                label = f"{gtype}_n{n}_i{inst}"
                records.append((label, n, srate, best_e, ground_e))

    labels = np.array([r[0] for r in records])
    n_labels = np.array([r[1] for r in records])
    success_rates = np.array([r[2] for r in records])
    best_energies = np.array([r[3] for r in records])
    ground_energies = np.array([r[4] for r in records])

    np.savez(
        HERE / "results.npz",
        labels=labels,
        n_labels=n_labels,
        success_rates=success_rates,
        best_energies=best_energies,
        ground_energies=ground_energies,
    )

if __name__ == "__main__":
    run()
