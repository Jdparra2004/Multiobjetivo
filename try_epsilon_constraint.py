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
def objective_flow(x):
    """Maximizar flujo total (convertido a minimización)"""
    return -np.sum(x)

def objective_cost(x):
    """Minimizar costo total"""
    flows = x.reshape((3, 2, 3))
    return np.sum(flows * C_ik[:, None, :]) + np.sum((flows / M) * C_ij[:, :, None])

def objective_emissions(x):
    """Minimizar emisiones totales"""
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
def optimize_with_epsilon_constraint(main_obj, epsilon_values, other_obj_bounds):
    """
    Optimiza con método ε-constraint
    
    Args:
        main_obj: Función objetivo principal a optimizar
        epsilon_values: Lista de valores ε para otros objetivos
        other_obj_bounds: Tupla con (min, max) de los otros objetivos
    """
    results = []
    x0 = np.full(n_vars, 100_000)  # Punto inicial
    
    # Generar valores ε para cada objetivo secundario
    eps_cost = np.linspace(other_obj_bounds[0][0], other_obj_bounds[0][1], epsilon_values)
    eps_emissions = np.linspace(other_obj_bounds[1][0], other_obj_bounds[1][1], epsilon_values)
    
    # Optimizar para cada combinación de ε
    for i, (eps_c, eps_e) in enumerate(zip(eps_cost, eps_emissions)):
        print(f"\nOptimizando con ε-constraints: Costo ≤ {eps_c:,.2f}, Emisiones ≤ {eps_e:,.2f}")
        
        # Añadir restricciones ε
        additional_constraints = [
            {'type': 'ineq', 'fun': lambda x: eps_c - objective_cost(x)},  # Costo ≤ ε_c
            {'type': 'ineq', 'fun': lambda x: eps_e - objective_emissions(x)}  # Emisiones ≤ ε_e
        ]
        
        all_constraints = constraints + additional_constraints
        
        res = minimize(
            main_obj,
            x0,
            method='SLSQP',
            bounds=bounds,
            constraints=all_constraints,
            options={'maxiter': 1000, 'ftol': 1e-6}
        )
        
        if res.success:
            flow = -main_obj(res.x) if main_obj == objective_flow else np.sum(res.x)
            cost = objective_cost(res.x)
            emissions = objective_emissions(res.x)
            results.append((flow, cost, emissions, res.x))
            
            print(f"  → Solución encontrada:")
            print(f"     Flujo: {flow:,.2f} kg")
            print(f"     Costo: ${cost:,.2f} COP")
            print(f"     Emisiones: {emissions:,.2f} kg CO₂")
        else:
            print(f"  → No se encontró solución factible para estos ε-constraints")
    
    return np.array(results, dtype=object)

# =============================
# 5. EJECUCIÓN DE LOS TRES ESCENARIOS ε-CONSTRAINT
# =============================
# Primero necesitamos estimar los rangos de los objetivos
print("\nEstimando rangos de objetivos para definir ε-constraints...")

# Optimizar cada objetivo individualmente para obtener límites
print("\n1. Optimizando para máximo flujo...")
res_flow = minimize(
    objective_flow,
    np.full(n_vars, 100_000),
    method='SLSQP',
    bounds=bounds,
    constraints=constraints,
    options={'maxiter': 500, 'ftol': 1e-6}
)
max_flow = -res_flow.fun if res_flow.success else 35_000_000  # Valor por defecto si falla

print("\n2. Optimizando para mínimo costo...")
res_cost = minimize(
    objective_cost,
    np.full(n_vars, 100_000),
    method='SLSQP',
    bounds=bounds,
    constraints=constraints,
    options={'maxiter': 500, 'ftol': 1e-6}
)
min_cost = res_cost.fun if res_cost.success else 2_000_000_000  # Valor por defecto si falla
max_cost = objective_cost(res_flow.x)  # Costo en solución de máximo flujo

print("\n3. Optimizando para mínimas emisiones...")
res_emissions = minimize(
    objective_emissions,
    np.full(n_vars, 100_000),
    method='SLSQP',
    bounds=bounds,
    constraints=constraints,
    options={'maxiter': 500, 'ftol': 1e-6}
)
min_emissions = res_emissions.fun if res_emissions.success else 5_000_000  # Valor por defecto si falla
max_emissions = objective_emissions(res_flow.x)  # Emisiones en solución de máximo flujo

print("\nRangos estimados:")
print(f"- Flujo: {F_min:,.0f} - {max_flow:,.0f} kg")
print(f"- Costo: ${min_cost:,.0f} - ${max_cost:,.0f} COP")
print(f"- Emisiones: {min_emissions:,.0f} - {max_emissions:,.0f} kg CO₂")

# Definir número de puntos ε
n_epsilon = 15

# Escenario 1: Maximizar flujo con ε en costo y emisiones
print("\n\nESCENARIO 1: Maximizar flujo con ε-constraints en costo y emisiones")
results_flow = optimize_with_epsilon_constraint(
    main_obj=objective_flow,
    epsilon_values=n_epsilon,
    other_obj_bounds=[(min_cost, max_cost), (min_emissions, max_emissions)]
)

# Escenario 2: Minimizar costo con ε en flujo y emisiones
print("\n\nESCENARIO 2: Minimizar costo con ε-constraints en flujo y emisiones")
results_cost = optimize_with_epsilon_constraint(
    main_obj=objective_cost,
    epsilon_values=n_epsilon,
    other_obj_bounds=[(F_min, max_flow), (min_emissions, max_emissions)]
)

# Escenario 3: Minimizar emisiones con ε en flujo y costo
print("\n\nESCENARIO 3: Minimizar emisiones con ε-constraints en flujo y costo")
results_emissions = optimize_with_epsilon_constraint(
    main_obj=objective_emissions,
    epsilon_values=n_epsilon,
    other_obj_bounds=[(F_min, max_flow), (min_cost, max_cost)]
)

# Combinar todos los resultados
all_results = np.concatenate([results_flow, results_cost, results_emissions])
flows = all_results[:, 0].astype(float)
costs = all_results[:, 1].astype(float)
emissions = all_results[:, 2].astype(float)
solutions = np.vstack(all_results[:, 3])

# =============================
# 6. VISUALIZACIÓN DE RESULTADOS ε-CONSTRAINT
# =============================
print("\nVISUALIZANDO RESULTADOS DEL MÉTODO ε-CONSTRAINT")

# 6.1. Gráfico 3D de todas las soluciones
fig = plt.figure(figsize=(12, 9))
ax = fig.add_subplot(111, projection='3d')

# Color por escenario
colors = ['blue']*len(results_flow) + ['green']*len(results_cost) + ['red']*len(results_emissions)
ax.scatter(flows, costs, emissions, c=colors, s=40, alpha=0.7)

ax.set_xlabel('Flujo Total (kg)', fontsize=12)
ax.set_ylabel('Costo Total (COP)', fontsize=12)
ax.set_zlabel('Emisiones CO₂ (kg)', fontsize=12)
ax.set_title('Soluciones por Método ε-Constraint\nAzul:MaxFlujo, Verde:MinCosto, Rojo:MinEmisiones', fontsize=14)
plt.tight_layout()
plt.savefig('epsilon_constraint_3d.png', dpi=300)
plt.show()

# 6.2. Proyecciones 2D
fig, ax = plt.subplots(1, 3, figsize=(18, 6))

# Flujo vs Costo
ax[0].scatter(results_flow[:, 0], results_flow[:, 1], c='blue', alpha=0.7, label='Max Flujo')
ax[0].scatter(results_cost[:, 0], results_cost[:, 1], c='green', alpha=0.7, label='Min Costo')
ax[0].scatter(results_emissions[:, 0], results_emissions[:, 1], c='red', alpha=0.7, label='Min Emisiones')
ax[0].set_xlabel('Flujo Total (kg)')
ax[0].set_ylabel('Costo Total (COP)')
ax[0].set_title('Flujo vs Costo')
ax[0].grid(True)
ax[0].legend()

# Flujo vs Emisiones
ax[1].scatter(results_flow[:, 0], results_flow[:, 2], c='blue', alpha=0.7, label='Max Flujo')
ax[1].scatter(results_cost[:, 0], results_cost[:, 2], c='green', alpha=0.7, label='Min Costo')
ax[1].scatter(results_emissions[:, 0], results_emissions[:, 2], c='red', alpha=0.7, label='Min Emisiones')
ax[1].set_xlabel('Flujo Total (kg)')
ax[1].set_ylabel('Emisiones CO₂ (kg)')
ax[1].set_title('Flujo vs Emisiones')
ax[1].grid(True)
ax[1].legend()

# Costo vs Emisiones
ax[2].scatter(results_flow[:, 1], results_flow[:, 2], c='blue', alpha=0.7, label='Max Flujo')
ax[2].scatter(results_cost[:, 1], results_cost[:, 2], c='green', alpha=0.7, label='Min Costo')
ax[2].scatter(results_emissions[:, 1], results_emissions[:, 2], c='red', alpha=0.7, label='Min Emisiones')
ax[2].set_xlabel('Costo Total (COP)')
ax[2].set_ylabel('Emisiones CO₂ (kg)')
ax[2].set_title('Costo vs Emisiones')
ax[2].grid(True)
ax[2].legend()

plt.tight_layout()
plt.savefig('epsilon_constraint_2d.png', dpi=300)
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

# Identificar soluciones no dominadas (Pareto) de todos los resultados
objectives = np.column_stack([-flows, costs, emissions])  # Convertimos flujo a minimización
pareto_mask = is_pareto_efficient(objectives)
pareto_flows = flows[pareto_mask]
pareto_costs = costs[pareto_mask]
pareto_emissions = emissions[pareto_mask]
pareto_solutions = solutions[pareto_mask]

print(f"\nTotal de soluciones ε-constraint: {len(flows)}")
print(f"Soluciones Pareto-eficientes encontradas: {len(pareto_flows)}")

# Análisis de soluciones Pareto-eficientes
if len(pareto_flows) > 0:
    # Encontrar soluciones extremas
    min_cost_idx = np.argmin(pareto_costs)
    min_emissions_idx = np.argmin(pareto_emissions)
    max_flow_idx = np.argmax(pareto_flows)
    
    # Solución balanceada
    normalized = (objectives[pareto_mask] - objectives[pareto_mask].min(axis=0)) / \
                (objectives[pareto_mask].max(axis=0) - objectives[pareto_mask].min(axis=0))
    balanced_idx = np.argmin(np.linalg.norm(normalized - 0.5, axis=1))
    
    # Mostrar resultados
    print_solution_stats(pareto_solutions[min_cost_idx], "Mínimo Costo (Pareto)")
    plot_composition(pareto_solutions[min_cost_idx], "Mínimo Costo (Pareto)")
    
    print_solution_stats(pareto_solutions[min_emissions_idx], "Mínimas Emisiones (Pareto)")
    plot_composition(pareto_solutions[min_emissions_idx], "Mínimas Emisiones (Pareto)")
    
    print_solution_stats(pareto_solutions[max_flow_idx], "Máximo Flujo (Pareto)")
    plot_composition(pareto_solutions[max_flow_idx], "Máximo Flujo (Pareto)")
    
    print_solution_stats(pareto_solutions[balanced_idx], "Solución Balanceada (Pareto)")
    plot_composition(pareto_solutions[balanced_idx], "Solución Balanceada (Pareto)")
    
    # Visualizar flujos
    plot_flows(pareto_solutions[min_cost_idx], "Mínimo Costo (Pareto)")
    plot_flows(pareto_solutions[min_emissions_idx], "Mínimas Emisiones (Pareto)")
    plot_flows(pareto_solutions[max_flow_idx], "Máximo Flujo (Pareto)")
    plot_flows(pareto_solutions[balanced_idx], "Solución Balanceada (Pareto)")

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