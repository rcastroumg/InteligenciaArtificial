# Clasificador de Aprobación de Créditos Bancarios
# Ejercicio de Machine Learning - Clasificación Supervisada

# 1. IMPORTAR BIBLIOTECAS NECESARIAS
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score, roc_curve
from sklearn.impute import SimpleImputer
import warnings
warnings.filterwarnings('ignore')

# Configuración para visualizaciones
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

print("=== CLASIFICADOR DE APROBACIÓN DE CRÉDITOS BANCARIOS ===")
print("Objetivo: Predecir si una solicitud de crédito será aprobada o rechazada\n")

# 2. CARGAR Y EXPLORAR LOS DATOS
print("1. CARGA Y EXPLORACIÓN INICIAL DE DATOS")
print("-" * 50)

# Cargar el dataset real UCI Credit Approval desde archivo crx.data
try:
    # Definir los nombres de las columnas según la documentación UCI
    column_names = ['A1', 'A2', 'A3', 'A4', 'A5', 'A6', 'A7', 'A8', 
                   'A9', 'A10', 'A11', 'A12', 'A13', 'A14', 'A15']
    
    # Cargar datos desde archivo crx.data
    df = pd.read_csv('/mnt/d/Descargas/credit+approval/crx.data', header=None, names=column_names, na_values='?')
    print("Dataset cargado exitosamente desde crx.data")
    
except FileNotFoundError:
    print("Archivo crx.data no encontrado. Descargando desde UCI...")
    # Descargar automáticamente si no existe el archivo
    import urllib.request
    
    try:
        url = "https://archive.ics.uci.edu/ml/machine-learning-databases/credit-screening/crx.data"
        urllib.request.urlretrieve(url, 'crx.data')
        df = pd.read_csv('crx.data', header=None, names=column_names, na_values='?')
        print("Dataset descargado y cargado exitosamente")
    except:
        print("Error al descargar. Usando datos sintéticos como respaldo...")
        # Código de respaldo con datos sintéticos (versión simplificada)
        np.random.seed(42)
        n_samples = 690  # Tamaño real del dataset UCI
        
        data = {
            'A1': np.random.choice(['a', 'b'], n_samples, p=[0.6, 0.4]),
            'A2': np.random.normal(31.57, 11.96, n_samples),
            'A3': np.random.normal(4.76, 4.98, n_samples),
            'A4': np.random.choice(['u', 'y', 'l'], n_samples, p=[0.5, 0.3, 0.2]),
            'A5': np.random.choice(['g', 'p', 'gg'], n_samples, p=[0.5, 0.3, 0.2]),
            'A6': np.random.choice(['c', 'd', 'cc', 'i', 'j', 'k', 'm', 'r', 'q', 'w', 'x', 'e', 'aa', 'ff'], n_samples),
            'A7': np.random.choice(['v', 'h', 'bb', 'j', 'n', 'z', 'dd', 'ff', 'o'], n_samples),
            'A8': np.random.normal(2.22, 3.35, n_samples),
            'A9': np.random.choice(['t', 'f'], n_samples, p=[0.7, 0.3]),
            'A10': np.random.normal(2.4, 3.25, n_samples),
            'A11': np.random.choice(['t', 'f'], n_samples, p=[0.6, 0.4]),
            'A12': np.random.choice(['s', 'g', 'p'], n_samples, p=[0.5, 0.3, 0.2]),
            'A13': np.random.normal(184.01, 173.81, n_samples),
            'A14': np.random.normal(1017.39, 5210.10, n_samples),
            'A15': np.random.choice(['+', '-'], n_samples, p=[0.56, 0.44])
        }
        df = pd.DataFrame(data)

print(f"Dimensiones del dataset: {df.shape}")
print(f"Características: {df.shape[1]-1}")
print(f"Muestras: {df.shape[0]}")

# Información básica del dataset
print("\nPrimeras 5 filas:")
print(df.head())

print("\nInformación del dataset:")
print(df.info())

print("\nEstadísticas descriptivas:")
print(df.describe())

# 3. ANÁLISIS EXPLORATORIO DE DATOS
print("\n2. ANÁLISIS EXPLORATORIO DE DATOS")
print("-" * 50)

# Distribución de la variable objetivo
print("Distribución de aprobaciones:")
target_dist = df['A15'].value_counts()
print(target_dist)
print(f"Porcentaje de aprobación: {target_dist['+'] / len(df) * 100:.2f}%")

# Crear visualizaciones
fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# Gráfico 1: Distribución de la variable objetivo
axes[0,0].pie(target_dist.values, labels=['Rechazado (-)', 'Aprobado (+)'], autopct='%1.1f%%')
axes[0,0].set_title('Distribución de Aprobaciones de Crédito')

# Gráfico 2: Edad vs Aprobación
df.boxplot(column='A2', by='A15', ax=axes[0,1])
axes[0,1].set_title('Edad por Estado de Aprobación')
axes[0,1].set_xlabel('Estado de Aprobación')
axes[0,1].set_ylabel('Edad')

# Gráfico 3: Ingresos vs Aprobación
df.boxplot(column='A13', by='A15', ax=axes[1,0])
axes[1,0].set_title('Ingresos por Estado de Aprobación')
axes[1,0].set_xlabel('Estado de Aprobación')
axes[1,0].set_ylabel('Ingresos')

# Gráfico 4: Cantidad solicitada vs Aprobación
df.boxplot(column='A14', by='A15', ax=axes[1,1])
axes[1,1].set_title('Cantidad Solicitada por Estado de Aprobación')
axes[1,1].set_xlabel('Estado de Aprobación')
axes[1,1].set_ylabel('Cantidad Solicitada')

plt.tight_layout()
plt.show()

# 4. PREPROCESAMIENTO DE DATOS
print("\n3. PREPROCESAMIENTO DE DATOS")
print("-" * 50)

# Hacer una copia para preprocesamiento
df_processed = df.copy()

# Identificar automáticamente variables numéricas y categóricas del dataset real
numeric_cols = []
categorical_cols = []

for col in df_processed.columns[:-1]:  # Excluir la variable objetivo A15
    # Verificar si la columna contiene valores numéricos
    if df_processed[col].dtype in ['int64', 'float64']:
        numeric_cols.append(col)
    else:
        # Para variables que pueden ser numéricas pero están como object debido a valores faltantes
        try:
            pd.to_numeric(df_processed[col], errors='raise')
            numeric_cols.append(col)
        except:
            categorical_cols.append(col)
            
# Ajustar tipos de datos conocidos del dataset UCI Credit Approval
# A2, A3, A8, A11, A14, A15 son típicamente numéricas
known_numeric = ['A2', 'A3', 'A8', 'A11', 'A14']
known_categorical = ['A1', 'A4', 'A5', 'A6', 'A7', 'A9', 'A10', 'A12', 'A13']

# Convertir variables conocidas como numéricas
for col in known_numeric:
    if col in df_processed.columns and col not in numeric_cols:
        try:
            df_processed[col] = pd.to_numeric(df_processed[col], errors='coerce')
            if col in categorical_cols:
                categorical_cols.remove(col)
            numeric_cols.append(col)
        except:
            pass

# Asegurar que variables categóricas conocidas estén en la lista correcta
for col in known_categorical:
    if col in df_processed.columns and col not in categorical_cols:
        if col in numeric_cols:
            numeric_cols.remove(col)
        categorical_cols.append(col)

print(f"Variables numéricas: {numeric_cols}")
print(f"Variables categóricas: {categorical_cols}")

# Verificar valores faltantes (el dataset real tiene valores faltantes marcados como '?')
print(f"\nValores faltantes por columna:")
missing_values = df_processed.isnull().sum()
print(missing_values[missing_values > 0])

if missing_values.sum() == 0:
    print("No se encontraron valores faltantes en el dataset.")
else:
    print(f"Total de valores faltantes: {missing_values.sum()}")
    print(f"Porcentaje de datos faltantes: {(missing_values.sum() / (df_processed.shape[0] * df_processed.shape[1])) * 100:.2f}%")

# Imputar valores faltantes para variables numéricas
imputer_numeric = SimpleImputer(strategy='median')
df_processed[numeric_cols] = imputer_numeric.fit_transform(df_processed[numeric_cols])

# Codificar variables categóricas
label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    df_processed[col] = le.fit_transform(df_processed[col].astype(str))
    label_encoders[col] = le

# Codificar variable objetivo
target_encoder = LabelEncoder()
df_processed['A15'] = target_encoder.fit_transform(df_processed['A15'])

print("Preprocesamiento completado.")
print("Variables categóricas codificadas con LabelEncoder")
print("Valores faltantes imputados con la mediana")

# 5. DIVIDIR DATOS EN ENTRENAMIENTO Y PRUEBA
print("\n4. DIVISIÓN DE DATOS")
print("-" * 50)

# Separar características y variable objetivo
X = df_processed.drop('A15', axis=1)
y = df_processed['A15']

# División en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print(f"Tamaño conjunto entrenamiento: {X_train.shape[0]} muestras")
print(f"Tamaño conjunto prueba: {X_test.shape[0]} muestras")
print(f"Distribución en entrenamiento: {np.bincount(y_train)}")
print(f"Distribución en prueba: {np.bincount(y_test)}")

# Escalar características numéricas
scaler = StandardScaler()
X_train_scaled = X_train.copy()
X_test_scaled = X_test.copy()
X_train_scaled[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
X_test_scaled[numeric_cols] = scaler.transform(X_test[numeric_cols])

# 6. ENTRENAR MODELOS DE CLASIFICACIÓN
print("\n5. ENTRENAMIENTO DE MODELOS")
print("-" * 50)

# Definir modelos
models = {
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'K-Nearest Neighbors': KNeighborsClassifier(),
    'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000)
}

# Entrenar y evaluar modelos
results = {}

for name, model in models.items():
    print(f"\nEntrenando {name}...")
    
    # Usar datos escalados para KNN y Logistic Regression
    if name in ['K-Nearest Neighbors', 'Logistic Regression']:
        X_train_use = X_train_scaled
        X_test_use = X_test_scaled
    else:
        X_train_use = X_train
        X_test_use = X_test
    
    # Entrenar modelo
    model.fit(X_train_use, y_train)
    
    # Predicciones
    y_pred = model.predict(X_test_use)
    y_pred_proba = model.predict_proba(X_test_use)[:, 1] if hasattr(model, 'predict_proba') else None
    
    # Métricas
    accuracy = accuracy_score(y_test, y_pred)
    
    # Validación cruzada
    cv_scores = cross_val_score(model, X_train_use, y_train, cv=5, scoring='accuracy')
    
    results[name] = {
        'model': model,
        'accuracy': accuracy,
        'cv_mean': cv_scores.mean(),
        'cv_std': cv_scores.std(),
        'predictions': y_pred,
        'probabilities': y_pred_proba
    }
    
    print(f"Precisión en prueba: {accuracy:.4f}")
    print(f"Validación cruzada: {cv_scores.mean():.4f} (+/- {cv_scores.std()*2:.4f})")

# 7. EVALUACIÓN DETALLADA DE MODELOS
print("\n6. EVALUACIÓN DETALLADA DE MODELOS")
print("-" * 50)

# Crear visualización de comparación
fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# Gráfico de precisión
model_names = list(results.keys())
accuracies = [results[name]['accuracy'] for name in model_names]
cv_means = [results[name]['cv_mean'] for name in model_names]

x_pos = np.arange(len(model_names))
axes[0,0].bar(x_pos, accuracies, alpha=0.7, label='Precisión en Prueba')
axes[0,0].bar(x_pos, cv_means, alpha=0.7, label='Validación Cruzada')
axes[0,0].set_xlabel('Modelos')
axes[0,0].set_ylabel('Precisión')
axes[0,0].set_title('Comparación de Precisión de Modelos')
axes[0,0].set_xticks(x_pos)
axes[0,0].set_xticklabels(model_names, rotation=45)
axes[0,0].legend()

# Encontrar el mejor modelo
best_model_name = max(results.keys(), key=lambda x: results[x]['cv_mean'])
best_model = results[best_model_name]['model']
best_predictions = results[best_model_name]['predictions']

print(f"Mejor modelo: {best_model_name}")
print(f"Precisión: {results[best_model_name]['accuracy']:.4f}")

# Matriz de confusión del mejor modelo
cm = confusion_matrix(y_test, best_predictions)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0,1])
axes[0,1].set_title(f'Matriz de Confusión - {best_model_name}')
axes[0,1].set_xlabel('Predicción')
axes[0,1].set_ylabel('Valor Real')

# Reporte de clasificación detallado
print(f"\nReporte de clasificación para {best_model_name}:")
target_names = ['Rechazado (-)', 'Aprobado (+)']
print(classification_report(y_test, best_predictions, target_names=target_names))

# Curva ROC para modelos con probabilidades
axes[1,0].set_title('Curvas ROC')
for name in results.keys():
    if results[name]['probabilities'] is not None:
        fpr, tpr, _ = roc_curve(y_test, results[name]['probabilities'])
        auc = roc_auc_score(y_test, results[name]['probabilities'])
        axes[1,0].plot(fpr, tpr, label=f'{name} (AUC = {auc:.3f})')

axes[1,0].plot([0, 1], [0, 1], 'k--', label='Línea base')
axes[1,0].set_xlabel('Tasa de Falsos Positivos')
axes[1,0].set_ylabel('Tasa de Verdaderos Positivos')
axes[1,0].legend()

# Importancia de características (para Decision Tree)
if best_model_name == 'Decision Tree':
    feature_importance = best_model.feature_importances_
    feature_names = X.columns
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': feature_importance
    }).sort_values('importance', ascending=True)
    
    axes[1,1].barh(importance_df['feature'], importance_df['importance'])
    axes[1,1].set_title('Importancia de Características')
    axes[1,1].set_xlabel('Importancia')
else:
    axes[1,1].text(0.5, 0.5, 'Importancia de características\nsolo disponible para Decision Tree', 
                   ha='center', va='center', transform=axes[1,1].transAxes)

plt.tight_layout()
plt.show()

# 8. OPTIMIZACIÓN DE HIPERPARÁMETROS
print("\n7. OPTIMIZACIÓN DE HIPERPARÁMETROS")
print("-" * 50)

# Optimizar el mejor modelo
if best_model_name == 'Decision Tree':
    param_grid = {
        'max_depth': [3, 5, 7, 10, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }
    X_opt = X_train
elif best_model_name == 'K-Nearest Neighbors':
    param_grid = {
        'n_neighbors': [3, 5, 7, 9, 11],
        'weights': ['uniform', 'distance'],
        'metric': ['euclidean', 'manhattan']
    }
    X_opt = X_train_scaled
else:  # Logistic Regression
    param_grid = {
        'C': [0.1, 1, 10, 100],
        'penalty': ['l1', 'l2'],
        'solver': ['liblinear', 'saga']
    }
    X_opt = X_train_scaled

# Búsqueda de hiperparámetros
grid_search = GridSearchCV(
    models[best_model_name], 
    param_grid, 
    cv=5, 
    scoring='accuracy',
    n_jobs=-1
)

grid_search.fit(X_opt, y_train)

print(f"Mejores hiperparámetros para {best_model_name}:")
print(grid_search.best_params_)
print(f"Mejor puntuación CV: {grid_search.best_score_:.4f}")

# Modelo optimizado
optimized_model = grid_search.best_estimator_

# Usar datos apropiados para predicción
if best_model_name in ['K-Nearest Neighbors', 'Logistic Regression']:
    X_test_final = X_test_scaled
else:
    X_test_final = X_test

y_pred_optimized = optimized_model.predict(X_test_final)
accuracy_optimized = accuracy_score(y_test, y_pred_optimized)

print(f"Precisión del modelo optimizado: {accuracy_optimized:.4f}")
print(f"Mejora: {accuracy_optimized - results[best_model_name]['accuracy']:.4f}")

# 9. INTERPRETACIÓN Y JUSTIFICACIÓN
print("\n8. INTERPRETACIÓN Y JUSTIFICACIÓN DEL MODELO")
print("=" * 60)

print(f"MODELO SELECCIONADO: {best_model_name}")
print(f"PRECISIÓN FINAL: {accuracy_optimized:.4f}")

print("\nJUSTIFICACIÓN:")
if best_model_name == 'Decision Tree':
    print("- Árbol de Decisión seleccionado por su interpretabilidad")
    print("- Permite entender fácilmente las reglas de decisión")
    print("- Ideal para explicar a stakeholders por qué se aprueba o rechaza un crédito")
    print("- Maneja bien variables categóricas y numéricas")
elif best_model_name == 'K-Nearest Neighbors':
    print("- KNN seleccionado por su simplicidad y efectividad")
    print("- Funciona bien con patrones locales en los datos")
    print("- No asume distribución específica de los datos")
    print("- Robusto a outliers cuando se usa distancia apropiada")
else:  # Logistic Regression
    print("- Regresión Logística seleccionada por su robustez estadística")
    print("- Proporciona probabilidades interpretables")
    print("- Coeficientes indican importancia y dirección de variables")
    print("- Ampliamente usado y aceptado en el sector financiero")

print(f"\nFACTORES CLAVE IDENTIFICADOS:")
if best_model_name == 'Decision Tree' and hasattr(optimized_model, 'feature_importances_'):
    feature_importance = optimized_model.feature_importances_
    feature_names = X.columns
    top_features = pd.DataFrame({
        'feature': feature_names,
        'importance': feature_importance
    }).sort_values('importance', ascending=False).head(5)
    
    for idx, row in top_features.iterrows():
        print(f"- {row['feature']}: {row['importance']:.3f}")

print(f"\nRECOMENDACIONES:")
print("1. Implementar el modelo en un entorno de prueba")
print("2. Monitorear el rendimiento con datos nuevos")
print("3. Reentrenar periódicamente con datos actualizados")
print("4. Considerar factores éticos y de sesgo en las decisiones")
print("5. Mantener un proceso de revisión humana para casos límite")

# 10. FUNCIÓN PARA PREDICCIONES NUEVAS
def predict_credit_approval(model, scaler, label_encoders, target_encoder, 
                          age, debt, work_years, income, amount, 
                          gender='a', marital='u', bank_account='g'):
    """
    Función para predecir aprobación de crédito para nuevos solicitantes
    """
    # Crear diccionario con valores
    new_data = {
        'A1': gender,      # Género
        'A2': age,         # Edad
        'A3': debt,        # Deuda
        'A4': marital,     # Estado civil
        'A5': bank_account, # Cuenta bancaria
        'A6': 'c',         # Trabajo (valor por defecto)
        'A7': 'v',         # Tipo trabajo (valor por defecto)
        'A8': work_years,  # Años en trabajo
        'A9': 't',         # Propiedad (valor por defecto)
        'A10': 2.0,        # Crédito existente (valor por defecto)
        'A11': 't',        # Historial crediticio (valor por defecto)
        'A12': 's',        # Ciudadanía (valor por defecto)
        'A13': income,     # Ingresos
        'A14': amount      # Cantidad solicitada
    }
    
    # Convertir a DataFrame
    new_df = pd.DataFrame([new_data])
    
    # Aplicar mismas transformaciones
    for col in categorical_cols:
        if col in label_encoders:
            # Manejar valores no vistos
            try:
                new_df[col] = label_encoders[col].transform(new_df[col])
            except ValueError:
                # Usar valor más común si no se vió en entrenamiento
                new_df[col] = 0
    
    # Escalar si es necesario
    if best_model_name in ['K-Nearest Neighbors', 'Logistic Regression']:
        new_df[numeric_cols] = scaler.transform(new_df[numeric_cols])
    
    # Predecir
    prediction = model.predict(new_df)[0]
    if hasattr(model, 'predict_proba'):
        probability = model.predict_proba(new_df)[0]
        return target_encoder.inverse_transform([prediction])[0], probability
    else:
        return target_encoder.inverse_transform([prediction])[0], None

# Ejemplo de uso
print(f"\nEJEMPLO DE PREDICCIÓN:")
result, prob = predict_credit_approval(
    optimized_model, scaler, label_encoders, target_encoder,
    age=35, debt=2.5, work_years=5, income=300, amount=800
)
print(f"Solicitante: 35 años, deuda=2.5, experiencia=5 años, ingresos=300, solicita=800")
print(f"Predicción: {result}")
if prob is not None:
    print(f"Probabilidades: Rechazo={prob[0]:.3f}, Aprobación={prob[1]:.3f}")

print(f"\n{'='*60}")
print("PROYECTO COMPLETADO EXITOSAMENTE")
print("El modelo está listo para ser implementado en producción")
print(f"{'='*60}")