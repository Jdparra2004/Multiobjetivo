import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import pandas as pd

#%% Data base del paper

I, J, K = 3, 2, 3

fuentes = ['Medellín', 'Cali', 'Bogotá']
sumideros = ['Ibagué', 'Maceo']
productos = ['Plástico', 'Textil', 'Papel']

S_ik = np.array([
    [1506.00, 3405.67, 3323.28],
    [4395.03, 1185.41, 12850.98],
    [233.37, 138.41, 243.92]
])

eta = np.array([0.5, 0.4, 0.1])

E_k = np.array([200, 240, 100])  # Emisión por kg de producto k [kg CO2]

# Distancia fuente i a sumidero j [km]
d_ij = np.array([
    [416, 122],
    [275, 528],
    [201, 402]
])

# Factor de emisión [kg CO2/km]
E_CO2_km = 0.7249

# Emisión de transporte (E_ij) = distancia × E_CO2_km
E_ij = d_ij * E_CO2_km

M = 9000
F_min = 3_000_000

#%% FUNCIONES AUXILIARES

def idx(i, j, k):
    return i * J * K + j * K + k

def reshape_F(F_flat):
    return F_flat.reshape((I, J, K))

#%% FUNCIÓN OBJETIVO: EMISIONES

def emission_function(F_flat):
    total = 0.0
    for i in range(I):
        for j in range(J):
            for k in range(K):
                f = F_flat[idx(i, j, k)]
                total += E_k[k] * f + (f / M) * E_ij[i, j]
    return total


#%% RESTRICCIONES

def build_constraints():
    cons = []

    # 1. Disponibilidad
    for i in range(I):
        for k in range(K):
            def restr_disp(F, i=i, k=k):
                return S_ik[i, k] - sum(F[idx(i, j, k)] for j in range(J))
            cons.append({'type': 'ineq', 'fun': restr_disp})

    # 2. Composición fija
    for j in range(J):
        total_F_j = lambda F, j=j: sum(F[idx(i, j, k)] for i in range(I) for k in range(K))
        for p in range(K - 1):
            def restr_eta(F, j=j, p=p):
                comp_p = sum(F[idx(i, j, p)] for i in range(I))
                return comp_p - eta[p] * total_F_j(F, j)
            cons.append({'type': 'eq', 'fun': restr_eta})

    # 3. Flujo mínimo
    def restr_fmin(F):
        return np.sum(F) - F_min
    cons.append({'type': 'ineq', 'fun': restr_fmin})

    return cons

#%% OPTIMIZACIÓN

n_vars = I * J * K
x0 = np.full(n_vars, 100.0)
bounds = [(0, None)] * n_vars

result = minimize(
    fun=emission_function,
    x0=x0,
    bounds=bounds,
    constraints=build_constraints(),
    method='SLSQP',
    options={'disp': True, 'maxiter': 1000}
)


#%% RESULTADOS

F_opt = reshape_F(result.x)
F_total = np.sum(F_opt)
E_total = emission_function(result.x)

# Tabla de resultados
data = []
for i in range(I):
    for j in range(J):
        for k in range(K):
            flujo = F_opt[i, j, k]
            if flujo > 1e-2:
                data.append([fuentes[i], sumideros[j], productos[k], flujo])

df = pd.DataFrame(data, columns=['Fuente', 'Sumidero', 'Producto', 'Flujo [kg]'])

# Mostrar tabla
tabla = df.pivot_table(index=['Fuente', 'Sumidero'], columns='Producto', values='Flujo [kg]')
print("\n📊 Tabla de flujos óptimos por producto [kg]:")
print(tabla.round(2))

# Flujo por sumidero
print("\n📌 Flujo total hacia cada sumidero [kg]:")
for j in range(J):
    flujo_j = np.sum(F_opt[:, j, :])
    print(f"- {sumideros[j]}: {flujo_j:.2f} kg")

# Totales
print(f"\n✅ Flujo total global: {F_total:,.2f} kg")
print(f"🌱 Emisión total mínima: {E_total:,.2f} kg CO2")

# GRÁFICAS

for j in range(J):
    fig, ax = plt.subplots()
    ancho = 0.3
    indices = np.arange(K)

    for i in range(I):
        flujos = [F_opt[i, j, k] for k in range(K)]
        bars = ax.bar(indices + i * ancho, flujos, width=ancho, label=fuentes[i])
        for bar in bars:
            height = bar.get_height()
            if height > 1e-2:
                ax.text(bar.get_x() + bar.get_width()/2, height + 50, f'{height:.0f} kg',
                        ha='center', va='bottom', fontsize=9)

    ax.set_xticks(indices + ancho)
    ax.set_xticklabels(productos)
    ax.set_title(f'Flujo hacia {sumideros[j]} [kg]')
    ax.set_ylabel('Flujo por producto [kg]')
    ax.legend()
    plt.tight_layout()
    plt.show()
