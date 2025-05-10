import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.special as special
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
# Cargar datos y limpieza
df=sns.load_dataset('titanic')
df.head(20)
df = df[['pclass', 'age', 'fare', 'alive']].dropna()  # Elimina filas con NaN

# Variables
X = df[['pclass', 'age', 'fare']]
y = df['alive']

# Verificar balance de clases
print("Distribución de 'alive':\n", y.value_counts())

# Dividir datos (con semilla fija)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Modelo SIN escalado y con balanceo de clases
algoritmo = LogisticRegression(class_weight='balanced', max_iter=1000)
algoritmo.fit(X_train, y_train)

# Predicciones y matriz de confusión
y_pred = algoritmo.predict(X_test)
matriz = confusion_matrix(y_test, y_pred)

print("\nMatriz de confusión:")
print(matriz)

# Métricas adicionales
from sklearn.metrics import accuracy_score, precision_score, recall_score
print("\nExactitud:", accuracy_score(y_test, y_pred))
print("Precisión (clase 'sí'):", precision_score(y_test, y_pred, pos_label='yes'))  # Ajusta 'yes' según tus datos
print("Sensibilidad (clase 'sí'):", recall_score(y_test, y_pred, pos_label='yes'))