import numpy as np
import pyomo.environ as pyo
from pyomo.opt import SolverFactory
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import warnings

# =============================
# 1. DATOS DEL ESTUDIO DE CASO
# =============================
# Fuentes (i): 0=Medellín, 1=Cali, 2=Bogotá
# Destinos (j): 0=Ibagué, 1=Macaeo
# Productos (k): 0=plástico, 1=textil, 2=papel

# Capacidades S_ik [kg] (Tabla 1)
S_ik_data = {
    (0, 0): 1_093_540, (0, 1): 3_390_808, (0, 2): 1_162_721,
    (1, 0): 15_603_343, (1, 1): 9_468_448, (1, 2): 1_162_721,
    (2, 0): 743_932, (2, 1): 1_093_395, (2, 2): 1_162_721
}

# Costos de compra C_ik [$/kg] (Tabla 1)
C_ik_data = {
    (0, 0): 1291.75, (0, 1): 436.50, (0, 2): 612.00,
    (1, 0): 1064.38, (1, 1): 485.25, (1, 2): 434.00,
    (2, 0): 1267.63, (2, 1): 466.13, (2, 2): 517.88
}

# Costos de transporte C_ij [$] (Tabla 4)
C_ij_data = {
    (0, 0): 121476.80, (0, 1): 45366.34,
    (1, 0): 101076.20, (1, 1): 200257.61,
    (2, 0): 75214.76, (2, 1): 157165.18
}

# Parámetros (Tabla 6)
eta = [0.50, 0.40, 0.10]     # Composición fija
M = 9000                      # Capacidad camión [kg]
E_k = [200, 240, 100]         # Emisiones CO₂/kg producto
E_CO2_km = 0.7249             # Emisiones CO₂/km
F_min = 3_000_000             # Flujo mínimo [kg]

# Precios gasolina G_i [$/galón] (Tabla 5)
G_i = [9373.21, 9463.97, 9355.07]
G_C = 25  # Consumo camión [km/galón]

# Calcular distancias d_ij [km]
d_ij_data = {}
for i in range(3):
    for j in range(2):
        d_ij_data[(i, j)] = (C_ij_data[(i, j)] * G_C) / G_i[i]

# Calcular emisiones transporte E_ij [kg CO₂]
E_ij_data = {}
for i in range(3):
    for j in range(2):
        E_ij_data[(i, j)] = d_ij_data[(i, j)] * E_CO2_km

print("Distancias calculadas (d_ij) [km]:")
print(np.array([[d_ij_data[(i, j)] for j in range(2)] for i in range(3)]))
print("Emisiones por transporte (E_ij) [kg CO2]:")
print(np.array([[E_ij_data[(i, j)] for j in range(2)] for i in range(3)]))

# =============================
# 2. DEFINIR EL MODELO PYOMO
# =============================
def build_model(weights):
    model = pyo.ConcreteModel()

    # Conjuntos
    model.I = pyo.Set(initialize=range(3))  # Fuentes
    model.J = pyo.Set(initialize=range(2))  # Destinos
    model.K = pyo.Set(initialize=range(3))  # Productos

    # Parámetros
    model.S_ik = pyo.Param(model.I, model.K, initialize=S_ik_data)
    model.C_ik = pyo.Param(model.I, model.K, initialize=C_ik_data)
    model.C_ij = pyo.Param(model.I, model.J, initialize=C_ij_data)
    model.E_k = pyo.Param(model.K, initialize={k: E_k[k] for k in range(3)})
    model.E_ij = pyo.Param(model.I, model.J, initialize=E_ij_data)
    model.eta = pyo.Param(model.K, initialize={k: eta[k] for k in range(3)})
    model.M = pyo.Param(initialize=M)
    model.F_min = pyo.Param(initialize=F_min)

    # Variables
    model.F_ijk = pyo.Var(model.I, model.J, model.K, domain=pyo.NonNegativeReals)

    # =============================
    # 2.1. Funciones Objetivo como expresiones
    # =============================
    # Objetivo 1: Maximizar el flujo total (minimizar su negativo)
    model.total_flow_obj = pyo.Expression(
        expr=-sum(model.F_ijk[i, j, k] for i in model.I for j in model.J for k in model.K)
    )

    # Objetivo 2: Minimizar el costo total
    model.total_cost_obj = pyo.Expression(expr=
        sum(model.F_ijk[i, j, k] * model.C_ik[i, k] for i in model.I for j in model.J for k in model.K) +
        sum((sum(model.F_ijk[i, j, k] for k in model.K) / model.M) * model.C_ij[i, j]
            for i in model.I for j in model.J)
    )

    # Objetivo 3: Minimizar las emisiones totales
    model.total_emissions_obj = pyo.Expression(expr=
        sum(model.F_ijk[i, j, k] * model.E_k[k] for i in model.I for j in model.J for k in model.K) +
        sum((sum(model.F_ijk[i, j, k] for k in model.K) / model.M) * model.E_ij[i, j]
            for i in model.I for j in model.J)
    )

    # =============================
    # 2.2. Función Objetivo Combinada (Suma Ponderada)
    # =============================
    w1, w2, w3 = weights
    model.obj = pyo.Objective(expr=
        w1 * model.total_flow_obj +
        w2 * model.total_cost_obj +
        w3 * model.total_emissions_obj,
        sense=pyo.minimize
    )

    # =============================
    # 2.3. Restricciones
    # =============================
    # Restricción de capacidad por fuente y producto
    def capacity_constraint_rule(model, i, k):
        return sum(model.F_ijk[i, j, k] for j in model.J) <= model.S_ik[i, k]
    model.capacity_constraint = pyo.Constraint(model.I, model.K, rule=capacity_constraint_rule)

    # Restricción de composición fija por destino (exacta)
    def composition_constraint_rule(model, j, k):
        return sum(model.F_ijk[i, j, k] for i in model.I) == model.eta[k] * sum(model.F_ijk[i, j, p] for i in model.I for p in model.K)
    model.composition_constraint = pyo.Constraint(model.J, model.K, rule=composition_constraint_rule)
    
    # Restricción de flujo mínimo
    def min_flow_constraint_rule(model):
        return sum(model.F_ijk[i, j, k] for i in model.I for j in model.J for k in model.K) >= model.F_min
    model.min_flow_constraint = pyo.Constraint(rule=min_flow_constraint_rule)

    return model

# =============================
# 3. ENFOQUE SUMA PONDERADA CON PYOMO
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

# Generar diferentes combinaciones de pesos
n_points = 10
weights_list = []
for w1 in np.linspace(0, 1, n_points):
    for w2 in np.linspace(0, 1 - w1, n_points):
        w3 = 1 - w1 - w2
        weights_list.append((w1, w2, w3))

# Almacenar resultados
results = []

# Usar el solver GLPK, ya que es gratuito y de propósito general.
# Para problemas no lineales o más grandes, se recomienda Gurobi, CPLEX, etc.
solver = SolverFactory('ipopt') # Usamos IPOPT para problemas no lineales

# Optimizar para cada combinación de pesos
print("Iniciando optimización con Pyomo...")
for i, weights in enumerate(weights_list):
    # Omitir combinaciones donde algún peso es 0 y los demás también (evita divisiones por cero)
    if sum(weights) == 0:
        continue
    
    model = build_model(weights)
    
    # Suprimir warnings de Pyomo
    warnings.filterwarnings('ignore', category=UserWarning, module='pyomo')
    
    # Resolver el modelo
    try:
        results_pyo = solver.solve(model, tee=False)
        
        if (results_pyo.solver.status == pyo.SolverStatus.ok and
            results_pyo.solver.termination_condition == pyo.TerminationCondition.optimal):
            
            # Obtener valores de los objetivos
            flow_val = -pyo.value(model.total_flow_obj)
            cost_val = pyo.value(model.total_cost_obj)
            emissions_val = pyo.value(model.total_emissions_obj)
            
            # Almacenar la solución
            solution_vals = np.array([pyo.value(model.F_ijk[i, j, k]) 
                                    for i in model.I for j in model.J for k in model.K])
            
            results.append((flow_val, cost_val, emissions_val, solution_vals))

            if i % 50 == 0 or i == 0 or i == len(weights_list) - 1:
                print(f"Optimizando combinación {i+1}/{len(weights_list)}: pesos = {weights}")
                print(f"  Flujo total: {flow_val:.2f} kg")
                print(f"  Costo total: ${cost_val:,.2f} COP")
                print(f"  Emisiones totales: {emissions_val:,.2f} kg CO2")
                print()
        else:
            if i % 50 == 0:
                print(f"  Falló con pesos {weights}: {results_pyo.solver.termination_condition}")

    except Exception as e:
        if i % 50 == 0:
            print(f"  Error resolviendo para pesos {weights}: {e}")

# Convertir a arrays para el análisis y visualización
results = np.array(results, dtype=object)
if len(results) > 0:
    flows = results[:, 0].astype(float)
    costs = results[:, 1].astype(float)
    emissions = results[:, 2].astype(float)
    solutions = np.vstack(results[:, 3])

    print(f"\nTotal de soluciones obtenidas: {len(flows)}")

    # =============================
    # 4. VISUALIZACIÓN 3D
    # =============================
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')
    sc = ax.scatter(flows, costs, emissions, c=emissions, cmap='viridis', s=40)
    ax.set_xlabel('Flujo Total (kg)')
    ax.set_ylabel('Costo Total (COP)')
    ax.set_zlabel('Emisiones CO₂ (kg)')
    ax.set_title('Espacio de Soluciones - Sin Filtrar (Pyomo)')
    fig.colorbar(sc, label='Emisiones CO₂ (kg)')
    plt.tight_layout()
    plt.savefig('espacio_pyomo_sin_filtrar.png', dpi=300)
    plt.show()

    # =============================
    # 5. FILTRAR FRENTE DE PARETO
    # =============================
    objectives = np.column_stack([-flows, costs, emissions])
    pareto_mask = is_pareto_efficient(objectives)
    pareto_flows = flows[pareto_mask]
    pareto_costs = costs[pareto_mask]
    pareto_emissions = emissions[pareto_mask]
    pareto_solutions = solutions[pareto_mask]

    print(f"\nTotal de soluciones Pareto-eficientes: {np.sum(pareto_mask)}")

    # =============================
    # 6. VISUALIZACIÓN DE TODAS LAS SOLUCIONES CON FRENTE DE PARETO
    # =============================
    fig, ax = plt.subplots(1, 3, figsize=(18, 6))
    
    # Proyecciones 2D
    ax[0].scatter(flows, costs, c='gray', alpha=0.5, s=30, label='Todas')
    ax[0].scatter(pareto_flows, pareto_costs, c='blue', s=40, label='Pareto')
    ax[0].set_xlabel('Flujo Total (kg)')
    ax[0].set_ylabel('Costo Total (COP)')
    ax[0].set_title('Flujo vs Costo (Pyomo)')
    ax[0].legend()

    ax[1].scatter(flows, emissions, c='gray', alpha=0.5, s=30, label='Todas')
    ax[1].scatter(pareto_flows, pareto_emissions, c='blue', s=40, label='Pareto')
    ax[1].set_xlabel('Flujo Total (kg)')
    ax[1].set_ylabel('Emisiones CO₂ (kg)')
    ax[1].set_title('Flujo vs Emisiones (Pyomo)')
    ax[1].legend()

    ax[2].scatter(costs, emissions, c='gray', alpha=0.5, s=30, label='Todas')
    ax[2].scatter(pareto_costs, pareto_emissions, c='blue', s=40, label='Pareto')
    ax[2].set_xlabel('Costo Total (COP)')
    ax[2].set_ylabel('Emisiones CO₂ (kg)')
    ax[2].set_title('Costo vs Emisiones (Pyomo)')
    ax[2].legend()

    plt.tight_layout()
    plt.savefig('todas_soluciones_pyomo_2d.png', dpi=300)
    plt.show()

else:
    print("No se encontraron soluciones viables. Revise las restricciones y el modelo.")