import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import pandas as pd

#%% Data base del Paper

# Número de fuentes, sumideros y productos
I, J, K = 3, 2, 3

# Disponibilidad por fuente y tipo de producto [kg]
S_ik = np.array([
    [1506.00, 3405.67, 3323.28],   # Medellín
    [4395.03, 1185.41, 12850.98],  # Cali
    [233.37, 138.41, 243.92]       # Bogotá
])

# Composición fija del combustible (η)
eta = np.array([0.5, 0.4, 0.1])  # [plástico, textil, papel]

#%% Funciones auxiliares para índices y restricciones
def idx(i, j, k):
    """Devuelve el índice plano para F_{ijk}."""
    return i * J * K + j * K + k

def reshape_F(F_flat):
    """Convierte el vector plano a una matriz [I, J, K]."""
    return F_flat.reshape((I, J, K))

#%% Función objetivo - Minimizar el negativo del flujo total, para así obtener el flujo máximo
def objective(F_flat):
    return -np.sum(F_flat)  # Maximizar flujo total

#%% Restricciones
def constraints():
    cons = []

    # Restricción 1: Disponibilidad en cada fuente para cada producto
    for i in range(I):
        for k in range(K):
            cons.append({
                'type': 'ineq',
                'fun': lambda F, i=i, k=k: S_ik[i, k] - sum(F[idx(i, j, k)] for j in range(J))
            })

    # Restricción 2: Composición fija en cada sumidero
    for j in range(J):
        total_flow = lambda F: sum(F[idx(i, j, k)] for i in range(I) for k in range(K))
        for p in range(K - 1):  # Solo K-1 restricciones independientes
            left = lambda F, j=j, p=p: sum(F[idx(i, j, p)] for i in range(I))
            right = lambda F, j=j, p=p: eta[p] * total_flow(F)
            cons.append({
                'type': 'eq',
                'fun': lambda F, j=j, p=p: left(F, j, p) - right(F, j, p)
            })

    return cons

#%% Configuración para el optimizador
# Número total de variables
n_vars = I * J * K

# Cotas: Todas las variables F_{ijk} ≥ 0
bounds = [(0, None) for _ in range(n_vars)]

# Valor inicial
x0 = np.full(n_vars, 100.0)

#Ejecutar solver del problema
result = minimize(
    fun=objective,
    x0=x0,
    bounds=bounds,
    constraints=constraints(),
    method='SLSQP',
    options={'disp': True, 'maxiter': 1000}
)

#%% Visualización de resultados
# Matriz F[i, j, k] con los flujos óptimos
F_opt = reshape_F(result.x)

# Flujo total (positivo)
F_total = np.sum(F_opt)
print(f"Flujo total óptimo: {F_total:.2f} kg")


#%% Tabla por sumidero
# Crear dataframe de resultados
columns = ['Fuente', 'Sumidero', 'Producto', 'Flujo (kg)']
data = []

fuentes = ['Medellín', 'Cali', 'Bogotá']
sumideros = ['Ibagué', 'Maceo']
productos = ['Plástico', 'Textil', 'Papel']

for i in range(I):
    for j in range(J):
        for k in range(K):
            data.append([fuentes[i], sumideros[j], productos[k], F_opt[i, j, k]])

df = pd.DataFrame(data, columns=['Fuente', 'Sumidero', 'Producto', 'Flujo [kg]'])

# Mostrar como tabla pivoteada
tabla = df.pivot_table(index=['Fuente', 'Sumidero'], columns='Producto', values='Flujo [kg]')
print("\n📊 Tabla de flujos óptimos por producto [kg]:")
print(tabla.round(2))


#%% Gráfica por producto y sumidero
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


#%% Resultados finales de interés
# Flujo total hacia cada sumidero
print("\n Flujo total hacia cada sumidero [kg]:")
for j in range(J):
    flujo_j = np.sum(F_opt[:, j, :])
    print(f"- {sumideros[j]}: {flujo_j:.2f} kg")


# Flujo total global
print(f"\n Flujo total global óptimo (todos los sumideros): {F_total:.2f} kg")

