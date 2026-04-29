import numpy as np
from dataclasses import dataclass

@dataclass
class Solution: 
    x_min: np.ndarray 
    xs: np.ndarray
    energies: np.ndarray 
    accepts: np.ndarray
    rejects: np.ndarray
    """ 
    A container class for outputting the optimal solution to the given QUBO problem
    :param x_min: The bit vector corresponding to the minimum of the QUBO
    :param xs: The various bit vectors cycled through throughout the annealing process
    :param energies: An energy series for data analysis
    :param accepts: The cumulative number of accepts at each iteration
    :param rejects: The cumulative number of rejects at each iteration 
    """