import numpy as np
import matplotlib.pyplot as plt

# Aqui va la funcion de la cual aproximaremos sus derivadas
def f(x):
    return np.sin(x)  # Función ejemplo

# Esta es la derivada analitica o exacta, se incluye con la
# finalidad de comparar y medir errores
def df_analytical(x):
    return np.cos(x)  # Derivada analítica de f(x)

# Metodo diferencias hacia adelante
def forward_diff(f, x, h=0.1):
    return (f(x + h) - f(x)) / h

# Metodo diferencias hacia atras
def backward_diff(f, x, h=0.1):
    return (f(x) - f(x - h)) / h

# Metodo diferencias centradas
def central_diff(f, x, h=0.1):
    return (f(x + h) - f(x - h)) / (2*h)

# Rango de valores para evaluar
# Definimos un intervalo
a = 0.0
b = np.pi
x_vals = np.linspace(a, b, 100)
df_exact = df_analytical(x_vals)

# Aproximaciones numéricas
df_forward = forward_diff(f, x_vals)
df_backward = backward_diff(f, x_vals)
df_central = central_diff(f, x_vals)

# Errores
error_forward = np.abs(df_forward - df_exact)
error_backward = np.abs(df_backward - df_exact)
error_central = np.abs(df_central - df_exact)

# Graficar las derivadas
plt.figure(figsize=(10, 6))
plt.plot(x_vals, f(x_vals), '-', label='Funcion')
plt.plot(x_vals, df_exact, 'k-', label='Derivada Analítica')
plt.plot(x_vals, df_forward, 'r--', label='Hacia adelante')
plt.plot(x_vals, df_backward, 'g-.', label='Hacia atrás')
plt.plot(x_vals, df_central, 'b:', label='Centrada')
plt.xlabel('x')
plt.ylabel("Derivada")
plt.legend()
plt.title("Comparación de Métodos de Diferenciación Numérica")
plt.grid()
plt.savefig("diferenciacion_aproximaciones.png")
plt.show()

# Graficar los errores
plt.figure(figsize=(10, 6))
plt.plot(x_vals, error_forward, 'r--', label='Error Hacia adelante')
plt.plot(x_vals, error_backward, 'g-.', label='Error Hacia atrás')
plt.plot(x_vals, error_central, 'b:', label='Error Centrada')
plt.xlabel('x')
plt.ylabel("Error absoluto")
plt.legend()
plt.title("Errores en Diferenciación Numérica")
plt.grid()
plt.savefig("diferenciacion_errores.png")
plt.show()

"Ejercicio 2"
import numpy as np
import matplotlib.pyplot as plt

# Aqui va la funcion de la cual aproximaremos sus derivadas
def f(x):
    return np.exp(x)  # Función ejemplo

# Esta es la derivada analitica o exacta, se incluye con la
# finalidad de comparar y medir errores
def df_analytical(x):
    return np.exp(x)  # Derivada analítica de f(x)

# Metodo diferencias hacia adelante
def forward_diff(f, x, h=0.05):
    return (f(x + h) - f(x)) / h

# Metodo diferencias hacia atras
def backward_diff(f, x, h=0.05):
    return (f(x) - f(x - h)) / h

# Metodo diferencias centradas
def central_diff(f, x, h=0.05):
    return (f(x + h) - f(x - h)) / (2*h)

# Rango de valores para evaluar
# Definimos un intervalo
a = 0.0
b = 2.0
x_vals = np.linspace(a, b, 100)
df_exact = df_analytical(x_vals)

# Aproximaciones numéricas
df_forward = forward_diff(f, x_vals)
df_backward = backward_diff(f, x_vals)
df_central = central_diff(f, x_vals)

# Errores
error_forward = np.abs(df_forward - df_exact)
error_backward = np.abs(df_backward - df_exact)
error_central = np.abs(df_central - df_exact)

# Graficar las derivadas
plt.figure(figsize=(10, 6))
plt.plot(x_vals, f(x_vals), '-', label='Funcion')
plt.plot(x_vals, df_exact, 'k-', label='Derivada Analítica')
plt.plot(x_vals, df_forward, 'r--', label='Hacia adelante')
plt.plot(x_vals, df_backward, 'g-.', label='Hacia atrás')
plt.plot(x_vals, df_central, 'b:', label='Centrada')
plt.xlabel('x')
plt.ylabel("Derivada")
plt.legend()
plt.title("Comparación de Métodos de Diferenciación Numérica")
plt.grid()
plt.savefig("diferenciacion_aproximaciones.png")
plt.show()

# Graficar los errores
plt.figure(figsize=(10, 6))
plt.plot(x_vals, error_forward, 'r--', label='Error Hacia adelante')
plt.plot(x_vals, error_backward, 'g-.', label='Error Hacia atrás')
plt.plot(x_vals, error_central, 'b:', label='Error Centrada')
plt.xlabel('x')
plt.ylabel("Error absoluto")
plt.legend()
plt.title("Errores en Diferenciación Numérica")
plt.grid()
plt.savefig("diferenciacion_errores.png")
plt.show()

"Ejercicio 3"

import numpy as np
import matplotlib.pyplot as plt

# Aqui va la funcion de la cual aproximaremos sus derivadas
def f(x):
    return x*3-2*x*2+x  # Función ejemplo

# Esta es la derivada analitica o exacta, se incluye con la
# finalidad de comparar y medir errores
def df_analytical(x):
    return 3*x**2-4*x+1  # Derivada analítica de f(x)

# Metodo diferencias hacia adelante
def forward_diff(f, x, h=0.2):
    return (f(x + h) - f(x)) / h

# Metodo diferencias hacia atras
def backward_diff(f, x, h=0.2):
    return (f(x) - f(x - h)) / h

# Metodo diferencias centradas
def central_diff(f, x, h=0.2):
    return (f(x + h) - f(x - h)) / (2*h)

# Rango de valores para evaluar
# Definimos un intervalo
a = -1.0
b = 2.0
x_vals = np.linspace(a, b, 100)
df_exact = df_analytical(x_vals)

# Aproximaciones numéricas
df_forward = forward_diff(f, x_vals)
df_backward = backward_diff(f, x_vals)
df_central = central_diff(f, x_vals)

# Errores
error_forward = np.abs(df_forward - df_exact)
error_backward = np.abs(df_backward - df_exact)
error_central = np.abs(df_central - df_exact)

# Graficar las derivadas
plt.figure(figsize=(10, 6))
plt.plot(x_vals, f(x_vals), '-', label='Funcion')
plt.plot(x_vals, df_exact, 'k-', label='Derivada Analítica')
plt.plot(x_vals, df_forward, 'r--', label='Hacia adelante')
plt.plot(x_vals, df_backward, 'g-.', label='Hacia atrás')
plt.plot(x_vals, df_central, 'b:', label='Centrada')
plt.xlabel('x')
plt.ylabel("Derivada")
plt.legend()
plt.title("Comparación de Métodos de Diferenciación Numérica")
plt.grid()
plt.savefig("diferenciacion_aproximaciones.png")
plt.show()

# Graficar los errores
plt.figure(figsize=(10, 6))
plt.plot(x_vals, error_forward, 'r--', label='Error Hacia adelante')
plt.plot(x_vals, error_backward, 'g-.', label='Error Hacia atrás')
plt.plot(x_vals, error_central, 'b:', label='Error Centrada')
plt.xlabel('x')
plt.ylabel("Error absoluto")
plt.legend()
plt.title("Errores en Diferenciación Numérica")
plt.grid()
plt.savefig("diferenciacion_errores.png")
plt.show()
