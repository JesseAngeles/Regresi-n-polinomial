import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures, StandardScaler, RobustScaler
from sklearn.linear_model import LinearRegression, SGDRegressor
from sklearn.metrics import mean_squared_error

from tabulate import tabulate

def polynomial_ols(x_train, y_train, x_test, y_test, degree, scaler = None):
    # Crear características polinomiales
    poly = PolynomialFeatures(degree=degree)
    x_train_poly = poly.fit_transform(x_train)
    x_test_poly = poly.fit_transform(x_test)  # Cambiado a transform para evitar fugas
    
    
    if scaler == 1:
        scaler = StandardScaler()
        x_train_poly = scaler.fit_transform(x_train_poly)
        x_test_poly = scaler.fit_transform(x_test_poly)  # Cambiado a transform para consistencia
    elif scaler == 2:
        scaler = RobustScaler()
        x_train_poly = scaler.fit_transform(x_train_poly)
        x_test_poly = scaler.fit_transform(x_test_poly) 
    
    # Ajustar modelo
    model = LinearRegression()
    model.fit(x_train_poly, y_train)
    y_pred = model.predict(x_test_poly)
    
    # Calcular métricas
    r2 = model.score(x_test_poly, y_test)
    mse = mean_squared_error(y_test, y_pred)
    
    return mse, r2

# Leer csv
df = pd.read_csv('resources/cal_housing.csv')

x = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, shuffle=True, random_state=0)

resultados = []

mse, r2 = polynomial_ols(x_train, y_train, x_test, y_test, degree=1, scaler=0)
resultados.append([f"Regresión polinomial de grado 1 con OLS", f"{mse:.6e}", f"{r2:.6e}"])

for i in range(2, 4):
    for j in range (3):
        if j == 0:
            s = "sin escalamiento"
        elif j == 1:
            s = "con escalamiento estandar"
        else:
            s = "con escalamiento robusto"
        
        mse, r2 = polynomial_ols(x_train, y_train, x_test, y_test, degree=i, scaler=j)
        resultados.append([f"Regresión polinomial de grado {i} {s}", 
                           f"{mse:.6e}", 
                           f"{r2:.6e}"])
        
# Crear tabla
tabla = tabulate(resultados, headers=["Método", "MSE", "R2"], tablefmt="github")
print(tabla)