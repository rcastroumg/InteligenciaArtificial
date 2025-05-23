import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Cargar el dataset
# url = "https://archive.ics.uci.edu/ml/machine-learning-databases/credit-approval/crx.data"
url = "/mnt/d/Descargas/credit+approval/crx.data"
columnas = ['Genero', 'Edad', 'Deuda', 'Casado', 'BancoCliente', 'EducacionNivel',
            'Etnia', 'AnosEmpleado', 'PreviamenteEmpleado', 'Ciudadano', 'CodigoPostal',
            'Ingresos', 'Aprobacion']
df = pd.read_csv(url, header=None, names=columnas)

# Exploración inicial
print("Primeras filas del dataset:")
print(df.head())
print("\nInformación del dataset:")
print(df.info())
print("\nResumen estadístico:")
print(df.describe())
print("\nValores únicos por columna:")
for columna in df.columns:
    print(f"{columna}: {df[columna].unique()}")



# 2.1. Manejar valores faltantes
# Reemplazamos '?' con NaN para facilitar el manejo
df = df.replace('?', pd.NA)

# Imputación de valores faltantes (usaremos la moda para categóricas y la media para numéricas por simplicidad)
for columna in df.columns:
    if df[columna].dtype == 'object':
        df[columna].fillna(df[columna].mode()[0], inplace=True)
    else:
        df[columna].fillna(df[columna].mean(), inplace=True)

print("\nValores faltantes después de la imputación:")
print(df.isnull().sum())

# 2.2. Codificar variables categóricas
# Usaremos Label Encoding para las columnas binarias y One-Hot Encoding para las demás
for columna in ['Genero', 'Casado', 'BancoCliente', 'PreviamenteEmpleado', 'Ciudadano', 'Aprobacion']:
    le = LabelEncoder()
    df[columna] = le.fit_transform(df[columna])

df = pd.get_dummies(df, columns=['EducacionNivel', 'Etnia'], drop_first=True)

# 2.3. Escalar variables numéricas
columnas_numericas = ['Edad', 'Deuda', 'AnosEmpleado', 'Ingresos']
scaler = StandardScaler()
df[columnas_numericas] = scaler.fit_transform(df[columnas_numericas])

# Eliminar la columna 'CodigoPostal' por tener demasiados valores únicos
df = df.drop('CodigoPostal', axis=1)

print("\nDataset después del preprocesamiento:")
print(df.head())



X = df.drop('Aprobacion', axis=1)
y = df['Aprobacion']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

print(f"\nTamaño del conjunto de entrenamiento: {X_train.shape}")
print(f"Tamaño del conjunto de prueba: {X_test.shape}")



# Inicializar los modelos
dt_classifier = DecisionTreeClassifier(random_state=42)
knn_classifier = KNeighborsClassifier()
lr_classifier = LogisticRegression(random_state=42)

# Entrenar los modelos
dt_classifier.fit(X_train, y_train)
knn_classifier.fit(X_train, y_train)
lr_classifier.fit(X_train, y_train)

print("\nModelos entrenados.")