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
n_points = 10
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
print(f"Emisiones mínimas: {np.min(pareto_emissions):,.2f} kg CO^2")
print(f"Emisiones máximas: {np.max(pareto_emissions):,.2f} kg CO^2")

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
ax.set_zlabel('Emisiones CO^2 (kg)', fontsize=12)
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
ax[1].set_ylabel('Emisiones CO^2 (kg)')
ax[1].set_title('Flujo vs Emisiones (Todas las soluciones)')
ax[1].grid(True)
ax[1].legend()

# Costo vs Emisiones (todas las soluciones)
ax[2].scatter(costs, emissions, c='gray', alpha=0.5, s=30, label='Todas')
if len(pareto_flows) > 0:
    ax[2].scatter(pareto_costs, pareto_emissions, c='blue', s=40, label='Pareto')
ax[2].set_xlabel('Costo Total (COP)')
ax[2].set_ylabel('Emisiones CO^2 (kg)')
ax[2].set_title('Costo vs Emisiones (Todas las soluciones)')
ax[2].grid(True)
ax[2].legend()

plt.tight_layout()
plt.savefig('todas_soluciones_2d.png', dpi=300)
plt.show()

# =============================
# 7. ANÁLISIS DE SOLUCIONES CLAVE
# =============================
def print_solution_stats(x, title=""):
    """Imprime estadísticas detalladas de una solución"""
    flows = x.reshape((3, 2, 3))
    total_flow = np.sum(flows)
    total_cost_val = total_cost(x)
    total_emissions_val = total_emissions(x)
    
    print("\n" + "="*60)
    print(f"ANÁLISIS DETALLADO - {title.upper()}")
    print("="*60)
    print(f"Flujo total: {total_flow:,.2f} kg")
    print(f"Costo total: ${total_cost_val:,.2f} COP")
    print(f"Emisiones totales: {total_emissions_val:,.2f} kg CO^2")
    
    # Análisis por destino
    for j in range(2):
        total_j = np.sum(flows[:, j, :])
        print(f"\nDestino {j+1} - Total: {total_j:,.2f} kg")
        
        # Composición porcentual
        comp = [np.sum(flows[:, j, k]) / total_j for k in range(3)]
        print("  Composición:")
        print(f"Plástico: {comp[0]*100:.2f}%")
        print(f"Textil: {comp[1]*100:.2f}%")
        print(f"Papel: {comp[2]*100:.2f}%")
        
        # Contribución por fuente
        print("\nContribución por fuente:")
        for i in range(3):
            source_contrib = np.sum(flows[i, j, :]) / total_j * 100
            print(f"    Fuente {i+1}: {source_contrib:.2f}%")

    # Análisis por fuente
    print("\nUtilización de capacidades por fuente:")
    for i in range(3):
        utilization = [np.sum(flows[i, :, k]) / S_ik[i, k] * 100 for k in range(3)]
        print(f"  Fuente {i+1}:")
        print(f"Plástico: {utilization[0]:.2f}% de capacidad")
        print(f"Textil: {utilization[1]:.2f}% de capacidad")
        print(f"Papel: {utilization[2]:.2f}% de capacidad")

# Función para graficar composición
def plot_composition(x, title):
    """Grafica la composición por destino"""
    flows = x.reshape((3, 2, 3))
    fig, ax = plt.subplots(1, 2, figsize=(15, 6))
    
    productos = ['Plástico', 'Textil', 'Papel']
    colores = ['#FF6B6B', '#4ECDC4', '#FFD166']
    
    for j in range(2):
        # Datos para el gráfico de torta
        composicion = [np.sum(flows[:, j, k]) for k in range(3)]
        total = sum(composicion)
        
        # Gráfico de torta
        ax[j].pie(composicion, labels=productos, autopct=lambda p: f'{p:.1f}%\n({p*total/100:,.0f} kg)',
                colors=colores, startangle=90)
        ax[j].set_title(f'Composición en Destino {j+1}\nTotal: {total:,.0f} kg')
    
    plt.suptitle(f'Composición del Combustible - {title}', fontsize=14)
    plt.tight_layout()
    plt.savefig(f'composicion_{title.lower().replace(" ", "_")}.png', dpi=300)
    plt.show()

# Encontrar soluciones extremas y representativas
if len(pareto_flows) > 0:
    # Soluciones extremas
    min_cost_idx = np.argmin(pareto_costs)
    min_emissions_idx = np.argmin(pareto_emissions)
    max_flow_idx = np.argmax(pareto_flows)
    
    # Solución balanceada (punto medio)
    normalized = (objectives[pareto_mask] - objectives[pareto_mask].min(axis=0)) / \
                (objectives[pareto_mask].max(axis=0) - objectives[pareto_mask].min(axis=0))
    balanced_idx = np.argmin(np.linalg.norm(normalized - 0.5, axis=1))
    
    # Análisis de soluciones
    print_solution_stats(solutions[pareto_mask][min_cost_idx], "Mínimo Costo")
    plot_composition(solutions[pareto_mask][min_cost_idx], "Mínimo Costo")
    
    print_solution_stats(solutions[pareto_mask][min_emissions_idx], "Mínimas Emisiones")
    plot_composition(solutions[pareto_mask][min_emissions_idx], "Mínimas Emisiones")
    
    print_solution_stats(solutions[pareto_mask][max_flow_idx], "Máximo Flujo")
    plot_composition(solutions[pareto_mask][max_flow_idx], "Máximo Flujo")
    
    print_solution_stats(solutions[pareto_mask][balanced_idx], "Solución Balanceada")
    plot_composition(solutions[pareto_mask][balanced_idx], "Solución Balanceada")
    
    # Tabla comparativa
    print("\n" + "="*60)
    print("COMPARATIVA DE SOLUCIONES CLAVE")
    print("="*60)
    print(f"{'':<20} | {'Flujo (kg)':>15} | {'Costo (COP)':>15} | {'Emisiones (kg CO^2)':>20}")
    print("-"*80)
    print(f"{'Mínimo Costo':<20} | {pareto_costs[min_cost_idx]:>15,.0f} | {pareto_costs[min_cost_idx]:>15,.0f} | {pareto_emissions[min_cost_idx]:>20,.0f}")
    print(f"{'Mínimas Emisiones':<20} | {pareto_flows[min_emissions_idx]:>15,.0f} | {pareto_costs[min_emissions_idx]:>15,.0f} | {pareto_emissions[min_emissions_idx]:>20,.0f}")
    print(f"{'Máximo Flujo':<20} | {pareto_flows[max_flow_idx]:>15,.0f} | {pareto_costs[max_flow_idx]:>15,.0f} | {pareto_emissions[max_flow_idx]:>20,.0f}")
    print(f"{'Solución Balanceada':<20} | {pareto_flows[balanced_idx]:>15,.0f} | {pareto_costs[balanced_idx]:>15,.0f} | {pareto_emissions[balanced_idx]:>20,.0f}")
else:
    print("\nNo se encontraron soluciones Pareto-eficientes para analizar")

# =============================
# 8. VISUALIZACIÓN DE FLUJOS (Mejorada)
# =============================
def plot_flows(x, title):
    """Visualiza los flujos óptimos con más detalle"""
    flows = x.reshape((3, 2, 3))
    fig, ax = plt.subplots(2, 3, figsize=(18, 12))
    
    productos = ['Plástico', 'Textil', 'Papel']
    fuentes = ['Medellín', 'Cali', 'Bogotá']
    destinos = ['Ibagué', 'Macaeo']
    colores = ['#FF6B6B', '#4ECDC4', '#FFD166']
    
    # Gráfico por producto y destino
    for k in range(3):
        for j in range(2):
            ax[0, k].bar(fuentes, flows[:, j, k], bottom=(flows[:, j, k-1] if k>0 else 0),
                        color=colores[j], label=destinos[j] if k==0 else "")
            ax[0, k].set_title(f'Distribución de {productos[k]}')
            ax[0, k].set_ylabel('Cantidad (kg)')
            if k == 0:
                ax[0, k].legend()
            ax[0, k].grid(axis='y', alpha=0.5)
    
    # Gráfico por fuente y destino
    for i in range(3):
        for j in range(2):
            ax[1, i].bar(productos, flows[i, j], color=colores[j],
                        label=destinos[j] if i==0 else "")
            ax[1, i].set_title(f'Aporte de {fuentes[i]}')
            ax[1, i].set_ylabel('Cantidad (kg)')
            if i == 0:
                ax[1, i].legend()
            ax[1, i].grid(axis='y', alpha=0.5)
    
    plt.suptitle(f'Distribución Detallada de Flujos - {title}', fontsize=16)
    plt.tight_layout()
    plt.savefig(f'flujos_detallados_{title.lower().replace(" ", "_")}.png', dpi=300)
    plt.show()

# Graficar soluciones clave si existen
if len(pareto_flows) > 0:
    plot_flows(solutions[pareto_mask][min_cost_idx], "Mínimo Costo")
    plot_flows(solutions[pareto_mask][min_emissions_idx], "Mínimas Emisiones")
    plot_flows(solutions[pareto_mask][max_flow_idx], "Máximo Flujo")
    plot_flows(solutions[pareto_mask][balanced_idx], "Solución Balanceada")