import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, classification_report
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# 1. Cargar el dataset
df = pd.read_csv("bangladesh_student_performance.csv")

# 2. Convertir HSC_Result a clasificación binaria: 1 = Aprobado (>3.5), 0 = Reprobado
df['HSC_Result'] = (df['HSC_Result'] > 3.5).astype(int)

print("Distribución de clases:")
print(df['HSC_Result'].value_counts())
print(f"  0 = Reprobado | 1 = Aprobado\n")

# 3. Seleccionar características predictoras
columnas = ['Gender', 'Age', 'School_Type', 'Study_Hours_per_Week',
            'Attendance', 'Internet_Access', 'Private_Tuition',
            'Previous_GPA', 'SSC_Result']

X = df[columnas]
y = df['HSC_Result']

# 4. Separar columnas numéricas y categóricas
columnas_numericas = ['Age', 'Study_Hours_per_Week', 'Attendance', 'Previous_GPA', 'SSC_Result']
columnas_categoricas = ['Gender', 'School_Type', 'Internet_Access', 'Private_Tuition']

# 5. Pipeline de Preprocesamiento
# Nota: SVM ES muy sensible a la escala, StandardScaler es OBLIGATORIO
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), columnas_numericas),
        ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), columnas_categoricas)
    ])

# 6. Construir el modelo SVM base (kernel RBF, el más potente para datos no lineales)
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', SVC(
        kernel='rbf',           # Kernel Radial Basis Function: captura relaciones no lineales
        C=1.0,                  # Parámetro de regularización
        gamma='scale',          # Escala automática del kernel
        class_weight='balanced',# Manejo de desbalance de clases
        probability=True,       # Necesario para calcular ROC AUC (predict_proba)
        random_state=42
    ))
])

# 7. División de datos estratificada (misma estrategia que modelos anteriores)
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
)

model.fit(X_train, y_train)

# 8. Predicción y Métricas en VALIDACIÓN
print("=" * 50)
print("Métricas en VALIDACIÓN (Modelo Base):")
print("=" * 50)
y_val_pred = model.predict(X_val)
y_val_proba = model.predict_proba(X_val)[:, 1]

print(f"Accuracy: {accuracy_score(y_val, y_val_pred):.4f}")
print(f"F1-Score: {f1_score(y_val, y_val_pred):.4f}")
print(f"ROC AUC:  {roc_auc_score(y_val, y_val_proba):.4f}")
print("\nReporte de Clasificación:\n",
      classification_report(y_val, y_val_pred, target_names=['Reprobado', 'Aprobado']))

# 9. Evaluación final en TEST (Modelo Base)
print("=" * 50)
print("Métricas en TEST (Modelo Base):")
print("=" * 50)
y_test_pred = model.predict(X_test)
y_test_proba = model.predict_proba(X_test)[:, 1]

print(f"Accuracy: {accuracy_score(y_test, y_test_pred):.4f}")
print(f"F1-Score: {f1_score(y_test, y_test_pred):.4f}")
print(f"ROC AUC:  {roc_auc_score(y_test, y_test_proba):.4f}")
print("\nReporte de Clasificación:\n",
      classification_report(y_test, y_test_pred, target_names=['Reprobado', 'Aprobado']))

# 10. Comparación de Kernels
print("\n" + "=" * 50)
print("Comparación de Kernels SVM:")
print("=" * 50)

kernels = {
    'RBF':      SVC(kernel='rbf',    C=1.0, gamma='scale', class_weight='balanced', probability=True, random_state=42),
    'Lineal':   SVC(kernel='linear', C=1.0,                class_weight='balanced', probability=True, random_state=42),
    'Polinómico': SVC(kernel='poly', C=1.0, degree=3, gamma='scale', class_weight='balanced', probability=True, random_state=42),
}

resultados_kernels = []
for nombre, clf in kernels.items():
    pipe_k = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', clf)])
    pipe_k.fit(X_train, y_train)
    y_pred_k = pipe_k.predict(X_val)
    y_proba_k = pipe_k.predict_proba(X_val)[:, 1]
    resultados_kernels.append({
        'Kernel':    nombre,
        'Accuracy':  round(accuracy_score(y_val, y_pred_k), 4),
        'F1-Score':  round(f1_score(y_val, y_pred_k), 4),
        'ROC AUC':   round(roc_auc_score(y_val, y_proba_k), 4)
    })

df_kernels = pd.DataFrame(resultados_kernels).set_index('Kernel')
print(df_kernels.to_string())

# 11. Optimización de Hiperparámetros con GridSearchCV (kernel RBF)
print("\n" + "=" * 50)
print("Optimización de Hiperparámetros (GridSearchCV - RBF):")
print("=" * 50)

param_grid = {
    'classifier__C':     [0.1, 1, 10, 100],
    'classifier__gamma': ['scale', 'auto', 0.01, 0.001],
}

grid_search = GridSearchCV(
    Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', SVC(kernel='rbf', class_weight='balanced', probability=True, random_state=42))
    ]),
    param_grid,
    cv=5,
    scoring='roc_auc',
    n_jobs=-1,
    verbose=1
)

grid_search.fit(X_train, y_train)

print(f"\nMejores hiperparámetros: {grid_search.best_params_}")
print(f"Mejor ROC AUC (CV):     {grid_search.best_score_:.4f}")

# 12. Evaluación del mejor modelo en TEST
best_model = grid_search.best_estimator_
y_test_pred_best = best_model.predict(X_test)
y_test_proba_best = best_model.predict_proba(X_test)[:, 1]

print("\n--- Métricas del Mejor Modelo SVM en TEST ---")
print(f"Accuracy: {accuracy_score(y_test, y_test_pred_best):.4f}")
print(f"F1-Score: {f1_score(y_test, y_test_pred_best):.4f}")
print(f"ROC AUC:  {roc_auc_score(y_test, y_test_proba_best):.4f}")
print("\nReporte de Clasificación:\n",
      classification_report(y_test, y_test_pred_best, target_names=['Reprobado', 'Aprobado']))