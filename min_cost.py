import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import pandas as pd

#%% Data base del paper

#Capacidad del camión
M = 9000  # [kg]

#Costos del Producto Cik en [$/Kg]
C_ik = np.array([
    [1291.75, 436.50, 612.00],    # Medellín
    [1064.38, 485.25, 434.00],    # Cali
    [1267.63, 466.13, 517.88]     # Bogotá
])
