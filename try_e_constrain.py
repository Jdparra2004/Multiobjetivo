# ============================================
# MODELO MULTIOBJETIVO CON MÉTODO ε-CONSTRAINT
# Basado en el caso colombiano del artículo PDF
# ============================================

import numpy as np
from scipy.optimize import minimize, Bounds
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# =============================
# 1. DATOS DEL ESTUDIO DE CASO
# =============================
S_ik = np.array([
    [1_093_540, 3_390_808, 1_162_721],   # Medellín
    [15_603_343, 9_468_448, 1_162_721],  # Cali
    [743_932, 1_093_395, 1_162_721]      # Bogotá
])

C_ik = np.array([
    [1291.75, 436.50, 612.00],
    [1064.38, 485.25, 434.00],
    [1267.63, 466.13, 517.88]
])

C_ij = np.array([
    [121476.80, 45366.34],
    [101076.20, 200257.61],
    [75214.76, 157165.18]
])

eta = [0.50, 0.40, 0.10]   # Composición fija
M = 9000
E_k = [200, 240, 100]
E_CO2_km = 0.7249
G_i = [9373.21, 9463.97, 9355.07]
G_C = 25
F_min = 3_000_000

# =============================
# 2. CÁLCULO DE DISTANCIAS Y EMISIONES
# =============================
d_ij = (C_ij * G_C) / np.array(G_i)[:, None]
E_ij = d_ij * E_CO2_km

n_vars = 3 * 2 * 3
x0 = np.ones(n_vars) * 1e5
bounds = Bounds(0, np.inf)

# =============================
# 3. FUNCIONES OBJETIVO
# =============================
def total_flow(x):
    return np.sum(x)

def total_cost(x):
    x_reshaped = x.reshape((3, 2, 3))
    return np.sum(x_reshaped * C_ik[:, None, :]) + np.sum((x_reshaped / M) * C_ij[:, :, None])

def total_emissions(x):
    x_reshaped = x.reshape((3, 2, 3))
    return np.sum(x_reshaped * E_k) + np.sum((x_reshaped / M) * E_ij[:, :, None])

# =============================
# 4. RESTRICCIONES COMUNES
# =============================
def base_constraints():
    return [
        {'type': 'ineq', 'fun': lambda x: np.array([S_ik[i, k] - np.sum(x.reshape((3, 2, 3))[i, :, k])
                                                    for i in range(3) for k in range(3)])},
        {'type': 'eq', 'fun': lambda x: np.array([
            np.sum(x.reshape((3, 2, 3))[:, j, p]) - eta[p] * np.sum(x.reshape((3, 2, 3))[:, j, :])
            for j in range(2) for p in range(2)])}
    ]

# =============================
# 5. IMPLEMENTACIÓN ε-CONSTRAINT
# =============================
eps_flows = np.linspace(3e6, 20e6, 10)
eps_emissions = np.linspace(1e9, 1.2e10, 10)

results = []

for ef in eps_flows:
    for ee in eps_emissions:
        constr = base_constraints()
        constr.extend([
            {'type': 'ineq', 'fun': lambda x, ef=ef: total_flow(x) - ef},
            {'type': 'ineq', 'fun': lambda x, ee=ee: ee - total_emissions(x)}
        ])

        res = minimize(total_cost, x0, method='SLSQP', bounds=bounds, constraints=constr,
                       options={'disp': False, 'maxiter': 500})

        if res.success:
            flow = total_flow(res.x)
            cost = total_cost(res.x)
            emissions = total_emissions(res.x)
            results.append((flow, cost, emissions))

# =============================
# 6. VISUALIZACIÓN SIN FILTRADO
# =============================
res_array = np.array(results)
fig = plt.figure(figsize=(12, 9))
ax = fig.add_subplot(111, projection='3d')
sc = ax.scatter(res_array[:,0], res_array[:,1], res_array[:,2], c=res_array[:,2], cmap='viridis', s=50)

ax.set_xlabel('Flujo Total (kg)')
ax.set_ylabel('Costo Total (COP)')
ax.set_zlabel('Emisiones CO₂ (kg)')
ax.set_title('Frente de Pareto (ε-constraint, sin filtrar)')
fig.colorbar(sc, label='Emisiones CO₂')
plt.tight_layout()
plt.savefig('pareto_epsilon_full.png', dpi=300)
plt.show()
