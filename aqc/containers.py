from dataclasses import dataclass
from typing import Dict
import numpy as np 

@dataclass
class Query:
    """
    Query Class to handle algorithm parameters

    Q: QUBO matrix
    iterations: Number of adiabatic steps. Defaults to 100
    total_time: Total evolution time T. Must be large relative to 1/gap^2. Defaults to 10.0
    shots: Number of experiment shots on Aer. Defaults to 1024
    trotter_order: Suzuki-Trotter order (1=Lie-Trotter, 2+=Suzuki-Trotter). Defaults to 2
    trotter_reps: Number of Trotter repetitions per step. Defaults to 1
    """
    Q: np.ndarray
    iterations: int = 100
    total_time: float = 10.0
    shots: int = 1024
    trotter_order: int = 2
    trotter_reps: int = 1
    schedule: str = 'linear'  # 'linear' | 'quadratic' | 'sinusoidal'

@dataclass
class Result:
    """
    Result Class to handle algorithm output

    prob: Probability of each bit vector
    eigs: Instantaneous eigenvalue series throughout evolution
    counts: Experiment count results
    statevector: Final statevector
    """
    prob: Dict
    eigs: np.ndarray
    counts: dict
    statevector: np.ndarray
