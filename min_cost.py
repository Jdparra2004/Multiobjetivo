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

# Composición fija [%]
eta = np.array([0.5, 0.4, 0.1])

# Costo de producto por kg [$]
C_ik = np.array([
    [1291.75, 436.50, 612.00],
    [1064.38, 485.25, 434.00],
    [1267.63, 466.13, 517.88]
])

# Costo de transporte [$]
C_ij = np.array([
    [121476.80, 45366.34],    # Medellín
    [101076.20, 200257.61],   # Cali
    [75214.76, 157165.18]     # Bogotá
])

# Capacidad del camión
M = 9000  # [kg]

# Flujo mínimo
F_min = 3_000_000  # [kg]

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

def constraints():
    cons = []

    # Restricción 1: disponibilidad
    for i in range(I):
        for k in range(K):
            cons.append({
                'type': 'ineq',
                'fun': lambda F, i=i, k=k: S_ik[i, k] - sum(F[idx(i, j, k)] for j in range(J))
            })

    # Restricción 2: composición fija
    for j in range(J):
        total_flow = lambda F: sum(F[idx(i, j, k)] for i in range(I) for k in range(K))
        for p in range(K - 1):  # solo K-1 restricciones independientes
            left = lambda F, j=j, p=p: sum(F[idx(i, j, p)] for i in range(I))
            right = lambda F, j=j, p=p: eta[p] * total_flow(F)
            cons.append({
                'type': 'eq',
                'fun': lambda F, j=j, p=p: left(F, j, p) - right(F, j, p)
            })

    return cons

# Añadimos la restricción del flujo mínimo
def constraints_scenario2():
    cons = constraints()
    cons.append({
        'type': 'ineq',
        'fun': lambda F: np.sum(F) - F_min
    })
    return cons

#%% Optimizador
n_vars = I * J * K
bounds = [(0, None) for _ in range(n_vars)]
x0 = np.full(n_vars, 100.0)

result2 = minimize(
    fun=cost_function,
    x0=x0,
    bounds=bounds,
    constraints=constraints_scenario2(),
    method='SLSQP',
    options={'disp': True, 'maxiter': 1000}
)

#%% Resultados}
F_opt2 = reshape_F(result2.x)
F_total2 = np.sum(F_opt2)
C_total2 = cost_function(result2.x)

# Tabla de resultados
data = []
for i in range(I):
    for j in range(J):
        for k in range(K):
            data.append([fuentes[i], sumideros[j], productos[k], F_opt2[i, j, k]])

df = pd.DataFrame(data, columns=['Fuente', 'Sumidero', 'Producto', 'Flujo [kg]'])
tabla = df.pivot_table(index=['Fuente', 'Sumidero'], columns='Producto', values='Flujo [kg]')

print("\n Tabla de flujos óptimos por producto [kg]:")
print(tabla.round(2))

# Flujo por sumidero
print("\n Flujo total hacia cada sumidero [kg]:")
for j in range(J):
    flujo_j = np.sum(F_opt2[:, j, :])
    print(f"- {sumideros[j]}: {flujo_j:.2f} kg")

print(f"\n Flujo total global: {F_total2:.2f} kg")
print(f" Costo total mínimo: {C_total2:,.2f} pesos colombianos")

#Gráficas
for j in range(J):
    fig, ax = plt.subplots()
    ancho = 0.3
    indices = np.arange(K)

    for i in range(I):
        flujos = [F_opt2[i, j, k] for k in range(K)]
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