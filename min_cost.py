import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import pandas as pd

#%% Data base del paper

I, J, K = 3, 2, 3  # fuentes, sumideros, productos

fuentes = ['Medellín', 'Cali', 'Bogotá']
sumideros = ['Ibagué', 'Maceo']
productos = ['Plástico', 'Textil', 'Papel']

# Disponibilidad por fuente y producto [kg]
S_ik = np.array([
    [1506.00, 3405.67, 3323.28],
    [4395.03, 1185.41, 12850.98],
    [233.37, 138.41, 243.92]
])

# Composición deseada del RDF (η)
eta = np.array([0.5, 0.4, 0.1])  # plástico, textil, papel

# Costo del producto por kg [$]
C_ik = np.array([
    [1291.75, 436.50, 612.00],
    [1064.38, 485.25, 434.00],
    [1267.63, 466.13, 517.88]
])

# Costo de transporte [$] desde i a j
C_ij = np.array([
    [121476.80, 45366.34],    # Medellín
    [101076.20, 200257.61],   # Cali
    [75214.76, 157165.18]     # Bogotá
])

# Parámetros adicionales
M = 9000              # Capacidad camión [kg]
F_min = 3_000_000     # Flujo mínimo [kg]

#%%  Funciones auxiliares
def idx(i, j, k):
    return i * J * K + j * K + k

def reshape_F(F_flat):
    return F_flat.reshape((I, J, K))

#%% Función objetivo (costo total)
def cost_function(F_flat):
    total = 0.0
    for i in range(I):
        for j in range(J):
            for k in range(K):
                f = F_flat[idx(i, j, k)]
                total += C_ik[i, k] * f + (f / M) * C_ij[i, j]
    return total

#%% Restricciones

def build_constraints():
    cons = []

    # Restricción 1: Disponibilidad por fuente y producto
    for i in range(I):
        for k in range(K):
            def restr_disp(F, i=i, k=k):
                return S_ik[i, k] - sum(F[idx(i, j, k)] for j in range(J))
            cons.append({'type': 'ineq', 'fun': restr_disp})

    # Restricción 2: Composición exacta por sumidero
    for j in range(J):
        total_F_j = lambda F, j=j: sum(F[idx(i, j, k)] for i in range(I) for k in range(K))
        for p in range(K - 1):  # Solo K-1 restricciones independientes
            def restr_eta(F, j=j, p=p):
                comp_p = sum(F[idx(i, j, p)] for i in range(I))
                return comp_p - eta[p] * total_F_j(F, j)
            cons.append({'type': 'eq', 'fun': restr_eta})

    # Restricción 3: Flujo total mínimo
    def restr_fmin(F):
        return np.sum(F) - F_min
    cons.append({'type': 'ineq', 'fun': restr_fmin})

    return cons

#%% Optimizador
n_vars = I * J * K
x0 = np.full(n_vars, 100.0)
bounds = [(0, None)] * n_vars

result = minimize(
    fun=cost_function,
    x0=x0,
    bounds=bounds,
    constraints=build_constraints(),
    method='SLSQP',
    options={'disp': True, 'maxiter': 1000}
)
#%% Resultados}
F_opt = reshape_F(result.x)
F_total = np.sum(F_opt)
C_total = cost_function(result.x)

# Tabla de resultados
data = []
for i in range(I):
    for j in range(J):
        for k in range(K):
            flujo = F_opt[i, j, k]
            if flujo > 1e-2:  # Mostrar solo flujos significativos
                data.append([fuentes[i], sumideros[j], productos[k], flujo])

df = pd.DataFrame(data, columns=['Fuente', 'Sumidero', 'Producto', 'Flujo [kg]'])

# Mostrar tabla de flujos
tabla = df.pivot_table(index=['Fuente', 'Sumidero'], columns='Producto', values='Flujo [kg]')
print("\n Tabla de flujos óptimos por producto [kg]:")
print(tabla.round(2))

# Mostrar flujos por sumidero
print("\n Flujo total hacia cada sumidero [kg]:")
for j in range(J):
    flujo_j = np.sum(F_opt[:, j, :])
    print(f"- {sumideros[j]}: {flujo_j:.2f} kg")

# Flujo total y costo total
print(f"\n Flujo total global: {F_total:,.2f} kg")
print(f" Costo total mínimo: {C_total:,.2f} pesos colombianos")

#Gráficas
for j in range(J):
    fig, ax = plt.subplots()
    ancho = 0.3
    indices = np.arange(K)

    for i in range(I):
        flujos = [F_opt[i, j, k] for k in range(K)]
        bars = ax.bar(indices + i * ancho, flujos, width=ancho, label=fuentes[i])
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, height + 50, f'{height:.0f} kg',
                    ha='center', va='bottom', fontsize=9)

    ax.set_xticks(indices + ancho)
    ax.set_xticklabels(productos)
    ax.set_title(f'Flujo hacia {sumideros[j]} [kg]')
    ax.set_ylabel('Flujo por producto [kg]')
    ax.legend()
    plt.tight_layout()
    plt.show()