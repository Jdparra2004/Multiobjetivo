import numpy as np
from scipy.optimize import minimize, Bounds
import matplotlib.pyplot as plt

# =============================
# 1. DATOS DEL ESTUDIO DE CASO
# =============================
# Fuentes (i): 0=Medellín, 1=Cali, 2=Bogotá
# Destinos (j): 0=Ibagué, 1=Macaeo
# Productos (k): 0=plástico, 1=textil, 2=papel

# Capacidades S_ik [kg] (Tabla 1 - Capacidad)
S_ik = np.array([
    [1_093_540, 3_390_808, 1_162_721],  # Medellín
    [15_603_343, 9_468_448, 1_162_721], # Cali
    [743_932, 1_093_395, 1_162_721]     # Bogotá
])

# Costos de compra C_ik [$/kg] (Tabla 1 - Precios)
C_ik = np.array([
    [1291.75, 436.50, 612.00],  # Medellín
    [1064.38, 485.25, 434.00],  # Cali
    [1267.63, 466.13, 517.88]   # Bogotá
])

# Costos de transporte C_ij [$] (Tabla 4)
C_ij = np.array([
    [121476.80, 45366.34],  # Medellín
    [101076.20, 200257.61], # Cali
    [75214.76, 157165.18]   # Bogotá
])

# Parámetros (Tabla 6)
eta = [0.50, 0.40, 0.10]    # Composición
M = 9000                     # Capacidad camión [kg]
E_k = [200, 240, 100]        # Emisiones CO₂/kg producto
E_CO2_km = 0.7249            # Emisiones CO₂/km
F_min = 3_000_000            # Flujo mínimo [kg] (Escenarios 2-3)

# Precios gasolina G_i [$/galón] (Tabla 5)
G_i = [9373.21, 9463.97, 9355.07]
G_C = 25  # Consumo camión [km/galón]

# Calcular emisiones transporte E_ij [kg CO₂] (Ecuación implícita)
d_ij = np.zeros_like(C_ij)
for i in range(3):
    for j in range(2):
        d_ij[i,j] = (C_ij[i,j] * G_C) / G_i[i]  # Distancia [km]
E_ij = d_ij * E_CO2_km  # Emisiones por viaje

# =============================
# 2. CONFIGURACIÓN OPTIMIZACIÓN
# =============================
n_vars = 3 * 2 * 3  # 18 variables (i,j,k)
bounds = Bounds(0, np.inf)  # F_ijk >= 0

# Restricción de capacidad (Ecuación 5)
def capacity_constraint(x):
    x = x.reshape((3,2,3))  # [i,j,k]
    constraints = []
    for i in range(3):
        for k in range(3):
            total_flow = sum(x[i,:,k])
            constraints.append(S_ik[i,k] - total_flow)
    return np.array(constraints)

# Restricción de composición (Ecuación 7)
def composition_constraint(x):
    x = x.reshape((3,2,3))
    constraints = []
    for j in range(2):
        total_flow_j = np.sum(x[:,j,:])
        for p in range(2):  # Solo primeros p-1 componentes
            component_flow = sum(x[:,j,p])
            constraints.append(component_flow - eta[p] * total_flow_j)
    return np.array(constraints)

# Restricción flujo mínimo (Ecuación 6)
def min_flow_constraint(x):
    return np.sum(x) - F_min

# Función para decodificar solución
def decode_solution(x):
    flows = x.reshape((3,2,3))
    total_flow = np.sum(flows)
    costs = np.sum(flows * C_ik[:,None,:]) + np.sum((flows/M) * C_ij[:,:,None])
    emissions = np.sum(flows * E_k) + np.sum((flows/M) * E_ij[:,:,None])
    return flows, total_flow, costs, emissions

# =============================
# 3. ESCENARIOS DE OPTIMIZACIÓN
# =============================
def run_scenario(scenario, use_min_flow=False):
    # Configurar restricciones
    constraints = [
        {'type': 'ineq', 'fun': capacity_constraint},
        {'type': 'eq', 'fun': composition_constraint}
    ]
    
    if use_min_flow:
        constraints.append({'type': 'ineq', 'fun': min_flow_constraint})
    
    # Función objetivo según escenario
    if scenario == 1:  # Maximizar flujo
        objective = lambda x: -np.sum(x)  # Negativo para maximizar
        x0 = np.full(n_vars, 100_000)    # Punto inicial
    elif scenario == 2:  # Minimizar costo
        def objective(x):
            flows = x.reshape((3,2,3))
            purchase_cost = np.sum(flows * C_ik[:,None,:])
            transport_cost = np.sum((flows/M) * C_ij[:,:,None])
            return purchase_cost + transport_cost
        x0 = np.full(n_vars, 50_000)
    else:  # Escenario 3: Minimizar emisiones
        def objective(x):
            flows = x.reshape((3,2,3))
            product_emissions = np.sum(flows * E_k)
            transport_emissions = np.sum((flows/M) * E_ij[:,:,None])
            return product_emissions + transport_emissions
        x0 = np.full(n_vars, 50_000)
    
    # Optimizar con SLSQP
    res = minimize(
        objective,
        x0,
        method='SLSQP',
        bounds=bounds,
        constraints=constraints,
        options={'maxiter': 500, 'ftol': 1e-6}
    )
    
    if not res.success:
        print(f"¡Advertencia! Optimización no convergió: {res.message}")
    
    return res

# =============================
# 4. EJECUCIÓN Y VISUALIZACIÓN
# =============================
def print_results(res, scenario):
    flows, total_flow, costs, emissions = decode_solution(res.x)
    
    print(f"\n{'='*50}")
    print(f"RESULTADOS ESCENARIO {scenario}")
    print(f"{'='*50}")
    print(f"• Flujo total: {total_flow:,.2f} kg")
    
    if scenario != 1:
        print(f"• Costo total: ${costs:,.2f} COP")
        print(f"• Emisiones totales: {emissions:,.2f} kg CO₂")
    
    # Imprimir flujos por origen-destino-producto
    print("\nFLUJOS ÓPTIMOS [kg]:")
    products = ['Plástico', 'Textil', 'Papel']
    sources = ['Medellín', 'Cali', 'Bogotá']
    destinations = ['Ibagué', 'Macaeo']
    
    for i in range(3):
        print(f"\nFuente: {sources[i]}")
        for j in range(2):
            print(f"  → Destino: {destinations[j]}")
            for k in range(3):
                print(f"    • {products[k]}: {flows[i,j,k]:>10,.2f} kg")

def plot_composition(flows, scenario):
    plt.figure(figsize=(10, 6))
    destinations = ['Ibagué', 'Macaeo']
    products = ['Plástico', 'Textil', 'Papel']
    colors = ['#FF6B6B', '#4ECDC4', '#FFD166']
    
    for j in range(2):
        plt.subplot(1, 2, j+1)
        comp = [np.sum(flows[:,j,k]) for k in range(3)]
        plt.pie(comp, labels=products, autopct='%1.1f%%', colors=colors)
        plt.title(f'Composición en {destinations[j]}\n(Escenario {scenario})')
    
    plt.tight_layout()
    plt.savefig(f'composicion_escenario_{scenario}.png')
    plt.show()

# Ejecutar todos los escenarios
for scenario in [1, 2, 3]:
    use_min_flow = (scenario != 1)
    res = run_scenario(scenario, use_min_flow)
    print_results(res, scenario)
    flows, _, _, _ = decode_solution(res.x)
    plot_composition(flows, scenario)