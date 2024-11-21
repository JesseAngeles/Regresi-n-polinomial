import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LinearRegression, SGDRegressor
from sklearn.metrics import mean_squared_error

from tabulate import tabulate

def polynomial_ols(x_train, y_train, x_test, y_test, degree):
    poly = PolynomialFeatures(degree=degree, include_bias=False)
    x_train_poly = poly.fit_transform(x_train)
    x_test_poly = poly.fit_transform(x_test)
    
    model = LinearRegression()
    model.fit(x_train_poly, y_train)
    y_pred = model.predict(x_test_poly)
    
    r2 = model.score(x_test_poly, y_test)
    mse = mean_squared_error(y_test, y_pred)
    
    sorted_indices = x_test.ravel().argsort()  # Ordenar x_test y usar los índices
    x_test_sorted = x_test[sorted_indices]
    y_test_sorted = y_test[sorted_indices]
    y_pred_sorted = y_pred[sorted_indices]
    
    plt.scatter(x_test_sorted, y_test_sorted, color='red', label='Puntos reales')
    plt.plot(x_test_sorted, y_pred_sorted, color='blue', label=f'Polinomio grado {degree}')
    
    # Personalización de la gráfica
    plt.title(f'Regresión Polinomial de grado {degree}')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    return mse, r2


def sgd(x_train, y_train, x_test, y_test, max_iter=10000, alpha=0.0000001):
    # Configurar y entrenar modelo SGD
    model = SGDRegressor(max_iter=max_iter, 
                         learning_rate='constant', 
                         alpha=alpha, 
                         random_state=0, 
                         eta0=alpha, 
                         shuffle=True,
                         tol=0.00112)
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)

    # Calcular métricas
    r2 = model.score(x_test, y_test)
    mse = mean_squared_error(y_test, y_pred)
    
    return r2, mse

def polynomial_sgd(x_train, y_train, x_test, y_test, degree, iterations=10000, alfa=0.0000001):
    # Generar características polinomiales
    poly = PolynomialFeatures(degree=degree, include_bias=True)
    x_train_poly = poly.fit_transform(x_train)
    x_test_poly = poly.fit_transform(x_test)  # Cambiado a transform

    # Configurar y entrenar modelo SGD
    model = SGDRegressor(
        max_iter=iterations,
        alpha=alfa,
        learning_rate='constant',
        eta0=alfa,
        tol=0.001119893383859,
        shuffle=True,
        random_state=0)
    
    model.fit(x_train_poly, y_train)
    y_pred = model.predict(x_test_poly)

    # Calcular métricas
    r2 = model.score(x_test_poly, y_test)
    mse = mean_squared_error(y_test, y_pred)

    sorted_indices = x_test.ravel().argsort()  # Ordenar x_test y usar los índices
    x_test_sorted = x_test[sorted_indices]
    y_test_sorted = y_test[sorted_indices]
    y_pred_sorted = y_pred[sorted_indices]
    
    plt.scatter(x_test_sorted, y_test_sorted, color='red', label='Puntos reales')
    plt.plot(x_test_sorted, y_pred_sorted, color='blue', label=f'Polinomio grado {degree}')
    
    # Personalización de la gráfica
    plt.title(f'Regresión Polinomial de grado {degree}')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend()
    plt.grid(True)
    plt.show()

    return mse, r2

# Leer csv
df = pd.read_csv('resources/datos.csv')

x = df['x'].values.reshape(-1, 1)  # Convertir a 2D
y = df['y'].values

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7, shuffle=True, random_state=0)

resultados = []

for i in range(1, 4):
    mse, r2 = polynomial_ols(x_train, y_train, x_test, y_test, degree=i)
    resultados.append([f"Regresión polinomial de grado {i} con OLS", f"{mse:.6e}", f"{r2:.6e}"])

# Calcular métricas para SGD
for i in range(1, 4):
    mse, r2 = polynomial_sgd(x_train, y_train, x_test, y_test, degree=i)
    resultados.append([f"Regresión polinomial de grado {i} con OLS", f"{mse:.6e}", f"{r2:.6e}"])

# Crear tabla
tabla = tabulate(resultados, headers=["Método", "MSE", "R2"], tablefmt="github")
print(tabla)