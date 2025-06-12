# Ver apuntes del 14/03/2025 En block de notas

'''
Esté código contempla un modelo no lineal, para el trabajo de minimización de costos
'''

import numpy as np
from scipy.optimize import minimize

# Capacidades de material disponible, ver tabla 1
capacidades = {
    "Medellín": {"Plástico": 1506, "Textil": 3405.67, "Papel": 3323.28},
    "Cali": {"Plástico": 4395.03, "Textil": 1185.41, "Papel": 12850.98},
    "Bogotá": {"Plástico": 233.37, "Textil": 138.41, "Papel": 243.92}
}

# Costo de transporte a cada refinería ver tabla 4
costos_transporte = {
    "Medellín": {"Ibagué": 121476.80, "Maceo": 45366.34},
    "Cali": {"Ibagué": 101076.20, "Maceo": 200257.61},
    "Bogotá": {"Ibagué": 75214.76, "Maceo": 157165.18}
}

# Proporciones requeridas de materiales en la mezcla final
proporciones = {"Plástico": 0.50, "Textil": 0.40, "Papel": 0.10}

# Definir función Objetivo
def costo_total(x):
    """
    Calcula el costo total de materiales y transporte.
    x: vector con las cantidades enviadas desde cada ciudad.
    """
    # Extraer las cantidades de cada material enviado desde cada ciudad
    x_med_plast, x_med_textil, x_med_papel = x[0], x[1], x[2]
    x_cal_plast, x_cal_textil, x_cal_papel = x[3], x[4], x[5]
    x_bog_plast, x_bog_textil, x_bog_papel = x[6], x[7], x[8]

    # Costo de transporte asociado
    costo_transporte = (
        (x_med_plast + x_med_textil + x_med_papel) * costos_transporte["Medellín"]["Ibagué"] +
        (x_cal_plast + x_cal_textil + x_cal_papel) * costos_transporte["Cali"]["Maceo"] +
        (x_bog_plast + x_bog_textil + x_bog_papel) * costos_transporte["Bogotá"]["Ibagué"]
    )

    # Retornar el costo total
    return costo_transporte

#definir las restricciones

def restricciones(x):
    """
    Define las restricciones de capacidad y composición.
    """
    # Suma total de materiales enviados
    total_material = sum(x)

    # Restricciones de capacidad (no se puede enviar más de lo disponible)
    restr_capacidad = [
        sum(x[:3]) - sum(capacidades["Medellín"].values()),  # Medellín
        sum(x[3:6]) - sum(capacidades["Cali"].values()),     # Cali
        sum(x[6:9]) - sum(capacidades["Bogotá"].values())    # Bogotá
    ]

    # Restricciones de composición (cumplir con las proporciones)
    restr_composicion = [
        (x[0] + x[3] + x[6]) / total_material - proporciones["Plástico"],  # Plástico
        (x[1] + x[4] + x[7]) / total_material - proporciones["Textil"],    # Textil
        (x[2] + x[5] + x[8]) / total_material - proporciones["Papel"]      # Papel
    ]

    # Devolvemos la lista de restricciones
    return restr_capacidad + restr_composicion

# --- Parámetros para la optimización ---
x0 = np.ones(9) * 500  # Valores iniciales para las cantidades
bounds = [(0, 5000)] * 9  # Límites para cada variable

# --- Resolver el problema de optimización ---
resultado = minimize(
    costo_total, x0, constraints={'type': 'eq', 'fun': restricciones}, bounds=bounds
)

# --- Imprimir los resultados ---
print("Solución óptima (cantidades enviadas desde cada ciudad):", resultado.x)
print("Costo mínimo encontrado:", resultado.fun)

print(resultado.success)