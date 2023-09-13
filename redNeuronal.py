#Importar librerias
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve
from sklearn.model_selection import cross_val_score
import seaborn as sns
import numpy as np


# Cargar el conjunto de datos iris.csv
file=input("Introduzca la ruta donde tiene gurdado el archivo: ")
data = pd.read_csv(file)

# Separar características (X) y etiquetas (y)
X = data.drop(['_id', 'Species'], axis=1)
y = data['Species']

# Codificar las etiquetas en números
"""Se utiliza la clase LabelEncoder de la biblioteca scikit-learn para convertir las 
etiquetas categóricas del conjunto de datos en valores numéricos. El método fit_transform ajusta el 
codificador a las etiquetas únicas en y y luego transforma y en una serie de números enteros que 
representan las etiquetas originales"""
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# Dividir el conjunto de datos en entrenamiento y prueba
"""Se utiliza la función train_test_split de scikit-learn para dividir el 
conjunto de datos en dos subconjuntos: uno para entrenamiento (X_train y y_train) y otro para 
prueba (X_test y y_test). La opción test_size especifica la proporción del conjunto de datos 
que se utilizará para pruebas"""

# Dividir el conjunto de datos en entrenamiento (80%) y prueba (20%)
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, random_state=42)

# Dividir el conjunto de datos de prueba en prueba (10%) y validación (10%)
X_test, X_validation, y_test, y_validation = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)


# Función para imprimir una matriz del conjunto de datos train, test y validation
def print_conjunto(matriz, titulo):
    print(titulo)
    matrix = np.array(matriz)
    print(matriz[:5])  # Imprimir solo las primeras 5 filas
    print("Shape:", matriz.shape)
    print()

# Imprimir matrices de 5 filas para los 3 conjuntos de datos
print_conjunto(X_train, "Conjunto de Entrenamiento (Train):")
print_conjunto(X_test, "Conjunto de prueba (Test):")
print_conjunto(X_validation, "Conjunto de validacion (Validation):")



# Crear y entrenar una red neuronal
"""Se utiliza MLPClassifier, que es un clasificador de perceptrón 
multicapa (red neuronal) de la biblioteca scikit-learn. Se configura con dos capas ocultas, 
cada una con 10 neuronas, el entrenamiento se detendrá después de un máximo de 1000 épocas. 
Luego, se entrena la red neuronal con los datos de entrenamiento (X_train y y_train) utilizando el método fit"""
clasificador_red = MLPClassifier(hidden_layer_sizes=(11, 11), max_iter=7000,alpha=0.5, random_state=42) #(10, 10) (11,11)
clasificador_red.fit(X_train, y_train)

# Realizar predicciones en el conjunto de prueba
"""Se utilizan los datos de prueba (X_test) para realizar predicciones utilizando 
la red neuronal entrenada. Las predicciones se almacenan en la variable y_pred, que contiene las 
etiquetas predichas para los ejemplos de prueba"""
y_pred = clasificador_red.predict(X_test)


#  evaluar sesgo y varianza
"""Obtener la curva de aprendizaje del modelo"""
train_sizes, train_scores, test_scores = learning_curve(
    clasificador_red, X, y, cv=5, scoring='accuracy', train_sizes=np.linspace(0.1, 1.0, 10))

# Estadisticas de desempeño del modelo
#Calcula la media de las puntuaciones de entrenamiento (train_scores) a lo largo del eje 1 (filas)
train_mean = np.mean(train_scores, axis=1)
# Calcula la desviación estándar de las puntuaciones de entrenamiento (train_scores) a lo largo del eje 1 (filas)
train_std = np.std(train_scores, axis=1)
# Calcula la media de las puntuaciones de validación cruzada (test_scores) a lo largo del eje 1 (filas)
test_mean = np.mean(test_scores, axis=1)
# Calcula la desviación estándar de las puntuaciones de validación cruzada (test_scores) a lo largo del eje 1 (filas)
test_std = np.std(test_scores, axis=1)


# Calcular las métricas de desempeño
"""Se utiliza la funcion classificatiom_report de scikit-learn para obtener todas las metricas del modelo como
precision, recall, f1score y accuracy"""
report=classification_report(y_test, y_pred)

# Calcular la matriz de confusión
"""con la funcion de confusion_matrix de scikit-learn obtenemos la matriz de confusion del modelo"""
confusion = confusion_matrix(y_test, y_pred)

# Mostrar los resultados
print("\nMatriz de Confusión:\n", confusion)
print("\nClassification report: ")
print(report)


# Calcula la puntuación de validación cruzada para evaluar el rendimiento del modelo
#Calcular bias y varianza
cv_scores = cross_val_score(clasificador_red, X, y, cv=5, scoring='accuracy')
bias = 1 - np.mean(cv_scores)
varianza = np.var(cv_scores)

# Imprime los resultados
print("Sesgo (Bias): {:.4f}".format(bias))
print("Varianza: {:.4f}".format(varianza))


# Encuentra el punto donde la diferencia entre train y test se minimiza
best_index = np.argmin(np.abs(train_mean - test_mean))

# Calcula el sesgo y la varianza en ese punto
bias_punto = 1 - train_mean[best_index]
varianza_punto = test_mean[best_index] - train_mean[best_index]

#declaramos que valores consideramos como bajo, medio o alto para la medicion de la varianza y el bias
def categorizar_medida(medida):
    if medida < 0.1:
        return "Bajo"
    elif medida < 0.3:
        return "Medio"
    else:
        return "Alto"

categor_bias = categorizar_medida(bias_punto)
categor_varianza = categorizar_medida(varianza_punto)

print("Calculo de bias y varianza en el punto de diferencia entre train y test: ")
print(f"Grado de Bias (Sesgo): {bias_punto:.4f} - {categor_bias}")
print(f"Grado de Varianza: {varianza_punto:.4f} - {categor_varianza}")


#comparaciones para determinar el nivel de ajuste del modelo (overfitting, fitt, underfitting)
if categor_bias == "Alto" and categor_varianza == "Alto":
    print("El modelo presenta alto bias y alta varianza por lo que existe underfitting y overfitting")
elif categor_bias == "Alto":
    print("El modelo presenta bias alto por lo que existe underfitting")
elif categor_varianza == "Alto":
    print("El modelo presenta varianza alta por lo que existe overfitting")
else:
    print("El modelo tiene buen equilibrio entre el bias y la varianza por lo que existe fitt")


"""# Crear un gráfico de calor (heatmap) de la matriz de confusión usando la funcion de seaborn de heatmap
plt.figure(figsize=(8, 6))
sns.set(font_scale=1.2)  
sns.heatmap(confusion, annot=True, fmt='g', cmap='Blues', cbar=False,
            xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
plt.xlabel('Predicciones')
plt.ylabel('Etiquetas verdaderas')
plt.title('Matriz de Confusión')
#plt.show()


# Plotea las curvas de aprendizaje para evaluar el sesgo (bias) en los conjuntos de datos
plt.figure(figsize=(10, 6))
plt.plot(train_sizes, train_mean, marker='o', label='Entrenamiento')
plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.15)
plt.plot(train_sizes, test_mean, marker='o', label='Prueba')
plt.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, alpha=0.15)
plt.xlabel('Tamaño del conjunto de entrenamiento')
plt.ylabel('Precisión')
plt.legend()
plt.title('Curvas de Sesgo (Bias) en los Conjuntos de Datos')
#plt.show()

# Plotea las curvas de aprendizaje para evaluar la varianza en los conjuntos de datos
plt.figure(figsize=(10, 6))
plt.plot(train_sizes, train_std, marker='o', label='Entrenamiento')
plt.plot(train_sizes, test_std, marker='o', label='Prueba')
plt.xlabel('Tamaño del conjunto de entrenamiento')
plt.ylabel('Desviación Estándar')
plt.legend()
plt.title('Curvas de Varianza en los Conjuntos de Datos')


# Plotea las curvas de sesgo (bias) y varianza en los conjuntos de datos
plt.figure(figsize=(10, 6))
plt.plot(train_sizes, train_mean, marker='o', label='Entrenamiento (Bias)')
plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.15)
plt.plot(train_sizes, test_std, marker='o', label='Prueba (Varianza)')
plt.xlabel('Tamaño del conjunto de entrenamiento')
plt.ylabel('Desempeño')
plt.legend()
plt.title('Nivel de Ajuste del Modelo (Bias vs. Varianza)')

plt.show()"""


# Crear una figura con subplots
fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# Gráfico de calor (Matriz de Confusión)
sns.set(font_scale=1.2)
sns.heatmap(confusion, annot=True, fmt='g', cmap='Blues', cbar=False,
            xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_, ax=axes[0, 0])
axes[0, 0].set_xlabel('Predicciones')
axes[0, 0].set_ylabel('Etiquetas verdaderas')
axes[0, 0].set_title('Matriz de Confusión')

# Curvas de Sesgo (Bias)
axes[0, 1].plot(train_sizes, train_mean, marker='o', label='Entrenamiento')
axes[0, 1].fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.15)
axes[0, 1].plot(train_sizes, test_mean, marker='o', label='Prueba')
axes[0, 1].fill_between(train_sizes, test_mean - test_std, test_mean + test_std, alpha=0.15)
axes[0, 1].set_xlabel('Tamaño del conjunto de entrenamiento')
axes[0, 1].set_ylabel('Precisión')
axes[0, 1].legend()
axes[0, 1].set_title('Curvas de Sesgo (Bias) en los Conjuntos de Datos')

# Curvas de Varianza
axes[1, 0].plot(train_sizes, train_std, marker='o', label='Entrenamiento')
axes[1, 0].plot(train_sizes, test_std, marker='o', label='Prueba')
axes[1, 0].set_xlabel('Tamaño del conjunto de entrenamiento')
axes[1, 0].set_ylabel('Desviación Estándar')
axes[1, 0].legend()
axes[1, 0].set_title('Curvas de Varianza en los Conjuntos de Datos')

# Nivel de Ajuste del Modelo (Bias vs. Varianza)
axes[1, 1].plot(train_sizes, train_mean, marker='o', label='Entrenamiento (Bias)')
axes[1, 1].fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.15)
axes[1, 1].plot(train_sizes, test_std, marker='o', label='Prueba (Varianza)')
axes[1, 1].set_xlabel('Tamaño del conjunto de entrenamiento')
axes[1, 1].set_ylabel('Desempeño')
axes[1, 1].legend()
axes[1, 1].set_title('Nivel de Ajuste del Modelo (Bias vs. Varianza)')

# Ajustar el espacio entre subplots
plt.tight_layout()

# Mostrar el plot
plt.show()


"""Mejoras usando Regularizacion"""
#En MLPClassifier de scikit-learn, se puede usar el parámetro alpha, 
# que controla la regularización L2 (también conocida como regularización de peso) en la red neuronal. 

# Crear y entrenar una red neuronal con regularización L2
clasificador_red_regularizado = MLPClassifier(hidden_layer_sizes=(2, 2), max_iter=7000, alpha=1, random_state=42)
clasificador_red_regularizado.fit(X_train, y_train)

# Realizar predicciones en el conjunto de prueba
y_pred_regularizado = clasificador_red_regularizado.predict(X_test)

# Calcular las métricas de desempeño con regularización
report_regularizado = classification_report(y_test, y_pred_regularizado)
confusion_regularizado = confusion_matrix(y_test, y_pred_regularizado)

# Mostrar los resultados con regularización
print("\nMatriz de Confusión con Regularización:\n", confusion_regularizado)
print("\nClassification report con Regularización: ")
print(report_regularizado)

