# ============================================
# MODELO MULTIOBJETIVO CON MÉTODO ε-CONSTRAINT
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
M = 9000  # Capacidad máxima del camión en kg
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
    def restric_capacidad_fuente(x):
        flows = x.reshape((3, 2, 3))
        return np.array([S_ik[i, k] - np.sum(flows[i, :, k]) for i in range(3) for k in range(3)])

    def restric_composicion_destino(x):
        flows = x.reshape((3, 2, 3))
        return np.array([
            np.sum(flows[:, j, p]) - eta[p] * np.sum(flows[:, j, :])
            for j in range(2) for p in range(2)
        ])

    def restric_carga_camion(x):
        flows = x.reshape((3, 2, 3))
        viajes_por_ruta = np.sum(flows, axis=2) / M  # i x j matriz de viajes
        return np.ravel(1e6 - viajes_por_ruta)  # Un número grande como umbral operativo

    return [
        {'type': 'ineq', 'fun': restric_capacidad_fuente},
        {'type': 'eq', 'fun': restric_composicion_destino},
        {'type': 'ineq', 'fun': restric_carga_camion}
    ]

# =============================
# 5. BÚSQUEDA ε-CONSTRAINT
# =============================
eps_flows = np.linspace(3e6, 20e6, 10)
eps_emissions = np.linspace(1e9, 1.2e10, 10)

results = []

for ef in eps_flows:
    for ee in eps_emissions:
        constraints = base_constraints() + [
            {'type': 'ineq', 'fun': lambda x, ef=ef: total_flow(x) - ef},
            {'type': 'ineq', 'fun': lambda x, ee=ee: ee - total_emissions(x)}
        ]

        res = minimize(total_cost, x0, method='SLSQP', bounds=bounds,
                    constraints=constraints, options={'disp': False})

        if res.success:
            f = total_flow(res.x)
            c = total_cost(res.x)
            e = total_emissions(res.x)
            results.append((f, c, e, res.x))
            print(f"Flujo: {f:,.2f} kg | Costo: ${c:,.2f} | Emisiones: {e:,.2f} kg CO^2")
        else:
            print(f"Fallo en epsilon_flujo={ef:.2e}, epsilon_emisiones={ee:.2e}")

# =============================
# 6. GRÁFICA 3D - TODAS LAS SOLUCIONES
# =============================
res_array = np.array(results, dtype=object)
flows = res_array[:, 0].astype(float)
costs = res_array[:, 1].astype(float)
emissions = res_array[:, 2].astype(float)

fig = plt.figure(figsize=(12, 9))
ax = fig.add_subplot(111, projection='3d')
sc = ax.scatter(flows, costs, emissions, c=emissions, cmap='viridis', s=50)

ax.set_xlabel('Flujo Total (kg)')
ax.set_ylabel('Costo Total (COP)')
ax.set_zlabel('Emisiones CO^2 (kg)')
ax.set_title('Frente de Pareto (epsilon_constraint, sin filtrar)')
fig.colorbar(sc, label='Emisiones CO^2')
plt.tight_layout()
plt.savefig('pareto_3d.png')
plt.show()

# =============================
# 7. GRÁFICAS 2D
# =============================
fig, ax = plt.subplots(1, 3, figsize=(18, 6))
ax[0].scatter(flows, costs, c='blue')
ax[0].set_xlabel('Flujo Total (kg)')
ax[0].set_ylabel('Costo Total (COP)')
ax[0].set_title('Flujo vs Costo')
ax[0].grid(True)

ax[1].scatter(flows, emissions, c='green')
ax[1].set_xlabel('Flujo Total (kg)')
ax[1].set_ylabel('Emisiones CO^2 (kg)')
ax[1].set_title('Flujo vs Emisiones')
ax[1].grid(True)

ax[2].scatter(costs, emissions, c='red')
ax[2].set_xlabel('Costo Total (COP)')
ax[2].set_ylabel('Emisiones CO^2 (kg)')
ax[2].set_title('Costo vs Emisiones')
ax[2].grid(True)

plt.tight_layout()
plt.savefig('pareto_2d.png')
plt.show()

# =============================
# 8. RESULTADOS DESTACADOS
# =============================
min_cost_idx = np.argmin(costs)
min_emis_idx = np.argmin(emissions)
max_flow_idx = np.argmax(flows)

print("\nMÍNIMO COSTO")
print(f"Flujo: {flows[min_cost_idx]:,.2f} kg")
print(f"Costo: ${costs[min_cost_idx]:,.2f}")
print(f"Emisiones: {emissions[min_cost_idx]:,.2f} kg CO^2")

print("\nMÍNIMAS EMISIONES")
print(f"Flujo: {flows[min_emis_idx]:,.2f} kg")
print(f"Costo: ${costs[min_emis_idx]:,.2f}")
print(f"Emisiones: {emissions[min_emis_idx]:,.2f} kg CO^2")

print("\nMÁXIMO FLUJO")
print(f"Flujo: {flows[max_flow_idx]:,.2f} kg")
print(f"Costo: ${costs[max_flow_idx]:,.2f}")
print(f"Emisiones: {emissions[max_flow_idx]:,.2f} kg CO^2")
