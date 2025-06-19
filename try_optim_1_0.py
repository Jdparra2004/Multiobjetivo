import numpy as np
from scipy.optimize import minimize, Bounds
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# =============================
# 1. DATOS DEL ESTUDIO DE CASO
# =============================
# Fuentes (i): 0=Medellín, 1=Cali, 2=Bogotá
# Destinos (j): 0=Ibagué, 1=Macaeo
# Productos (k): 0=plástico, 1=textil, 2=papel

# Capacidades S_ik [kg] (Tabla 1)
S_ik = np.array([
    [1_093_540, 3_390_808, 1_162_721],   # Medellín
    [15_603_343, 9_468_448, 1_162_721],  # Cali
    [743_932, 1_093_395, 1_162_721]      # Bogotá
])

# Costos de compra C_ik [$/kg] (Tabla 1)
C_ik = np.array([
    [1291.75, 436.50, 612.00],   # Medellín
    [1064.38, 485.25, 434.00],   # Cali
    [1267.63, 466.13, 517.88]    # Bogotá
])

# Costos de transporte C_ij [$] (Tabla 4)
C_ij = np.array([
    [121476.80, 45366.34],   # Medellín
    [101076.20, 200257.61],  # Cali
    [75214.76, 157165.18]    # Bogotá
])

# Parámetros (Tabla 6)
eta = [0.50, 0.40, 0.10]     # Composición fija
M = 9000                      # Capacidad camión [kg]
E_k = [200, 240, 100]         # Emisiones CO₂/kg producto
E_CO2_km = 0.7249             # Emisiones CO₂/km
F_min = 3_000_000             # Flujo mínimo [kg]

# Precios gasolina G_i [$/galón] (Tabla 5)
G_i = [9373.21, 9463.97, 9355.07]
G_C = 25  # Consumo camión [km/galón]

# Calcular distancias d_ij [km] (de Tabla 4)
d_ij = np.zeros_like(C_ij)
for i in range(3):
    for j in range(2):
        d_ij[i, j] = (C_ij[i, j] * G_C) / G_i[i]

# Calcular emisiones transporte E_ij [kg CO₂]
E_ij = d_ij * E_CO2_km

print("Distancias calculadas (d_ij) [km]:\n", d_ij)
print("Emisiones por transporte (E_ij) [kg CO2]:\n", E_ij)

# Número de variables (3 fuentes × 2 destinos × 3 productos)
n_vars = 3 * 2 * 3

# =============================
# 2. FUNCIONES OBJETIVO
# =============================
def total_flow(x):
    """Flujo total (maximizar)"""
    return -np.sum(x)  # Negativo para convertir en minimización

def total_cost(x):
    """Costo total (minimizar)"""
    flows = x.reshape((3, 2, 3))
    return np.sum(flows * C_ik[:, None, :]) + np.sum((flows / M) * C_ij[:, :, None])

def total_emissions(x):
    """Emisiones totales (minimizar)"""
    flows = x.reshape((3, 2, 3))
    return np.sum(flows * E_k) + np.sum((flows / M) * E_ij[:, :, None])

# =============================
# 3. RESTRICCIONES
# =============================
def capacity_constraint(x):
    """Restricción de capacidad por fuente y producto"""
    flows = x.reshape((3, 2, 3))
    constraints = []
    for i in range(3):
        for k in range(3):
            total_flow = np.sum(flows[i, :, k])
            constraints.append(S_ik[i, k] - total_flow)
    return np.array(constraints)

def composition_constraint(x):
    """Restricción de composición fija por destino"""
    flows = x.reshape((3, 2, 3))
    constraints = []
    for j in range(2):
        total_flow_j = np.sum(flows[:, j, :])
        for p in range(2):  # Solo primeros p-1 componentes
            component_flow = np.sum(flows[:, j, p])
            constraints.append(component_flow - eta[p] * total_flow_j)
    return np.array(constraints)

def min_flow_constraint(x):
    """Restricción de flujo mínimo"""
    return np.sum(x) - F_min

# Configurar restricciones para SciPy
constraints = [
    {'type': 'ineq', 'fun': capacity_constraint},  # S_ik - ΣF_ijk >= 0
    {'type': 'eq', 'fun': composition_constraint},  # Composición exacta
    {'type': 'ineq', 'fun': min_flow_constraint}    # Flujo total >= F_min
]

# Límites de variables (flujos no negativos)
bounds = Bounds(0, np.inf)  # F_ijk >= 0

# =============================
# 4. ENFOQUE SUMA PONDERADA
# =============================
def combined_objective(x, weights):
    """Función objetivo combinada con pesos"""
    w1, w2, w3 = weights
    return (
        w1 * total_flow(x) +
        w2 * total_cost(x) +
        w3 * total_emissions(x)
    )

# Generar diferentes combinaciones de pesos
n_points = 5
weights_list = []
for w1 in np.linspace(0, 1, n_points):
    for w2 in np.linspace(0, 1 - w1, n_points):
        w3 = 1 - w1 - w2
        weights_list.append((w1, w2, w3))

print(f"Número total de combinaciones de pesos: {len(weights_list)}")

# Almacenar resultados
results = []
x0 = np.full(n_vars, 100_000)  # Punto inicial

# Optimizar para cada combinación de pesos
for i, weights in enumerate(weights_list):
    if i % 50 == 0 or i == 0 or i == len(weights_list)-1:
        print(f"Optimizando combinación {i+1}/{len(weights_list)}: pesos = {weights}")
    
    res = minimize(
        combined_objective,
        x0,
        args=(weights,),
        method='SLSQP',
        bounds=bounds,
        constraints=constraints,
        options={'maxiter': 500, 'ftol': 1e-6}
    )
    
    if res.success:
        flow = -total_flow(res.x)  # Convertir a positivo
        cost = total_cost(res.x)
        emissions = total_emissions(res.x)
        results.append((flow, cost, emissions, res.x))
        
        # Mostrar resultados para esta combinación
        if i % 50 == 0 or i == 0 or i == len(weights_list)-1:
            print(f"  Flujo total: {flow:.2f} kg")
            print(f"  Costo total: ${cost:,.2f} COP")
            print(f"  Emisiones totales: {emissions:,.2f} kg CO2")
            print()
    else:
        print(f"  Falló con pesos {weights}: {res.message}")

# Convertir a arrays
results = np.array(results, dtype=object)
flows = results[:, 0].astype(float)
costs = results[:, 1].astype(float)
emissions = results[:, 2].astype(float)
solutions = np.vstack(results[:, 3])

print(f"\nTotal de soluciones obtenidas: {len(flows)}")

# =============================
# 4.1. Visualización 3D - TODAS LAS SOLUCIONES SIN FILTRAR
# =============================
fig = plt.figure(figsize=(12, 9))
ax = fig.add_subplot(111, projection='3d')

sc = ax.scatter(flows, costs, emissions, c=emissions, cmap='viridis', s=40)

ax.set_xlabel('Flujo Total (kg)', fontsize=12)
ax.set_ylabel('Costo Total (COP)', fontsize=12)
ax.set_zlabel('Emisiones CO₂ (kg)', fontsize=12)
ax.set_title('Espacio de Soluciones - Sin Filtrar', fontsize=14)
fig.colorbar(sc, label='Emisiones CO₂ (kg)')
plt.tight_layout()
plt.savefig('espacio_sin_filtrar.png', dpi=300)
plt.show()

# =============================
# 5. FILTRAR FRENTE DE PARETO
# =============================
def is_pareto_efficient(costs):
    """Identifica soluciones Pareto-eficientes"""
    is_efficient = np.ones(costs.shape[0], dtype=bool)
    for i, c in enumerate(costs):
        if is_efficient[i]:
            # Dominio: menor costo, menor emisión, mayor flujo
            is_efficient[is_efficient] = np.any(costs[is_efficient] < c, axis=1) 
            is_efficient[i] = True
    return is_efficient

# Juntar objetivos en una matriz
objectives = np.column_stack([-flows, costs, emissions])  # Convertimos flujo a minimización

# Encontrar soluciones Pareto-eficientes
pareto_mask = is_pareto_efficient(objectives)
pareto_flows = flows[pareto_mask]
pareto_costs = costs[pareto_mask]
pareto_emissions = emissions[pareto_mask]
pareto_solutions = solutions[pareto_mask]

print(f"\nTotal de soluciones Pareto-eficientes: {np.sum(pareto_mask)}")
print(f"Flujo mínimo: {np.min(pareto_flows):,.2f} kg")
print(f"Flujo máximo: {np.max(pareto_flows):,.2f} kg")
print(f"Costo mínimo: ${np.min(pareto_costs):,.2f}")
print(f"Costo máximo: ${np.max(pareto_costs):,.2f}")
print(f"Emisiones mínimas: {np.min(pareto_emissions):,.2f} kg CO2")
print(f"Emisiones máximas: {np.max(pareto_emissions):,.2f} kg CO2")

# =============================
# 6. VISUALIZACIÓN DE TODAS LAS SOLUCIONES (ANTES DE FILTRAR)
# =============================
print("\nVISUALIZANDO TODAS LAS SOLUCIONES OBTENIDAS (ANTES DE FILTRAR PARETO)")

# 6.1. Gráfico 3D de todas las soluciones
fig = plt.figure(figsize=(12, 9))
ax = fig.add_subplot(111, projection='3d')

# Graficar todas las soluciones (no filtradas)
sc_all = ax.scatter(flows, costs, emissions, 
                c='gray', alpha=0.5, s=30, 
                label='Todas las soluciones')

# Graficar soluciones Pareto-eficientes (si existen)
if len(pareto_flows) > 0:
    sc_pareto = ax.scatter(pareto_flows, pareto_costs, pareto_emissions, 
                        c='blue', s=50, 
                        label='Soluciones Pareto-eficientes')

ax.set_xlabel('Flujo Total (kg)', fontsize=12)
ax.set_ylabel('Costo Total (COP)', fontsize=12)
ax.set_zlabel('Emisiones CO₂ (kg)', fontsize=12)
ax.set_title('Todas las Soluciones Obtenidas', fontsize=14)
ax.legend()
plt.tight_layout()
plt.savefig('todas_soluciones_3d.png', dpi=300)
plt.show()

# 6.2. Proyecciones 2D de todas las soluciones
fig, ax = plt.subplots(1, 3, figsize=(18, 6))

# Flujo vs Costo (todas las soluciones)
ax[0].scatter(flows, costs, c='gray', alpha=0.5, s=30, label='Todas')
if len(pareto_flows) > 0:
    ax[0].scatter(pareto_flows, pareto_costs, c='blue', s=40, label='Pareto')
ax[0].set_xlabel('Flujo Total (kg)')
ax[0].set_ylabel('Costo Total (COP)')
ax[0].set_title('Flujo vs Costo (Todas las soluciones)')
ax[0].grid(True)
ax[0].legend()

# Flujo vs Emisiones (todas las soluciones)
ax[1].scatter(flows, emissions, c='gray', alpha=0.5, s=30, label='Todas')
if len(pareto_flows) > 0:
    ax[1].scatter(pareto_flows, pareto_emissions, c='blue', s=40, label='Pareto')
ax[1].set_xlabel('Flujo Total (kg)')
ax[1].set_ylabel('Emisiones CO₂ (kg)')
ax[1].set_title('Flujo vs Emisiones (Todas las soluciones)')
ax[1].grid(True)
ax[1].legend()

# Costo vs Emisiones (todas las soluciones)
ax[2].scatter(costs, emissions, c='gray', alpha=0.5, s=30, label='Todas')
if len(pareto_flows) > 0:
    ax[2].scatter(pareto_costs, pareto_emissions, c='blue', s=40, label='Pareto')
ax[2].set_xlabel('Costo Total (COP)')
ax[2].set_ylabel('Emisiones CO₂ (kg)')
ax[2].set_title('Costo vs Emisiones (Todas las soluciones)')
ax[2].grid(True)
ax[2].legend()

plt.tight_layout()
plt.savefig('todas_soluciones_2d.png', dpi=300)
plt.show()

# =============================
# 7. ANÁLISIS DE SOLUCIONES CLAVE
# =============================
def print_solution_stats(x):
    """Imprime estadísticas de una solución"""
    flows = x.reshape((3, 2, 3))
    total_flow = np.sum(flows)
    total_cost_val = total_cost(x)
    total_emissions_val = total_emissions(x)
    
    print(f"Flujo total: {total_flow:,.2f} kg")
    print(f"Costo total: ${total_cost_val:,.2f} COP")
    print(f"Emisiones totales: {total_emissions_val:,.2f} kg CO2")
    
    # Composición por destino
    for j in range(2):
        total_j = np.sum(flows[:, j, :])
        comp_plastic = np.sum(flows[:, j, 0]) / total_j
        comp_textil = np.sum(flows[:, j, 1]) / total_j
        comp_paper = np.sum(flows[:, j, 2]) / total_j
        
        print(f"\nDestino {j+1} - Composición:")
        print(f"  Plástico: {comp_plastic*100:.2f}%")
        print(f"  Textil: {comp_textil*100:.2f}%")
        print(f"  Papel: {comp_paper*100:.2f}%")

# Encontrar soluciones extremas
min_cost_idx = np.argmin(pareto_costs)
min_emissions_idx = np.argmin(pareto_emissions)
max_flow_idx = np.argmax(pareto_flows)

print("\n" + "="*50)
print("SOLUCIÓN DE MÍNIMO COSTO")
print("="*50)
print_solution_stats(pareto_solutions[min_cost_idx])

print("\n" + "="*50)
print("SOLUCIÓN DE MÍNIMAS EMISIONES")
print("="*50)
print_solution_stats(pareto_solutions[min_emissions_idx])

print("\n" + "="*50)
print("SOLUCIÓN DE MÁXIMO FLUJO")
print("="*50)
print_solution_stats(pareto_solutions[max_flow_idx])

# =============================
# 8. VISUALIZACIÓN DE FLUJOS
# =============================
def plot_flows(x, title):
    """Visualiza los flujos óptimos"""
    flows = x.reshape((3, 2, 3))
    fig, ax = plt.subplots(1, 2, figsize=(15, 6))
    
    destinos = ['Ibagué', 'Macaeo']
    productos = ['Plástico', 'Textil', 'Papel']
    colores = ['#FF6B6B', '#4ECDC4', '#FFD166']
    
    for j in range(2):
        bottom = np.zeros(3)
        for i in range(3):
            ax[j].bar(productos, flows[i, j], bottom=bottom, 
                    label=f'Fuente {i+1}', color=colores[i])
            bottom += flows[i, j]
        
        ax[j].set_title(f'Flujos a {destinos[j]}')
        ax[j].set_ylabel('Cantidad (kg)')
        ax[j].legend()
        ax[j].grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.suptitle(title, fontsize=14)
    plt.tight_layout()
    plt.savefig(f'flujos_{title.lower().replace(" ", "_")}.png', dpi=300)
    plt.show()

# Graficar soluciones clave
plot_flows(pareto_solutions[min_cost_idx], "Mínimo Costo")
plot_flows(pareto_solutions[min_emissions_idx], "Mínimas Emisiones")
plot_flows(pareto_solutions[max_flow_idx], "Máximo Flujo")