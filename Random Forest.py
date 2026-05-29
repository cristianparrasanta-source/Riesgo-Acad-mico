import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, classification_report
from sklearn.preprocessing import OneHotEncoder
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
# Nota: Random Forest NO requiere escalar variables numéricas
columnas_numericas = ['Age', 'Study_Hours_per_Week', 'Attendance', 'Previous_GPA', 'SSC_Result']
columnas_categoricas = ['Gender', 'School_Type', 'Internet_Access', 'Private_Tuition']

# 5. Pipeline de Preprocesamiento (sin StandardScaler, no es necesario para RF)
preprocessor = ColumnTransformer(
    transformers=[
        ('num', 'passthrough', columnas_numericas),  # RF no necesita escalar
        ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), columnas_categoricas)
    ])

# 6. Construir el modelo base
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        min_samples_split=5,
        min_samples_leaf=2,
        max_features='sqrt',
        class_weight='balanced',  # Manejo de desbalance de clases (igual que en RL)
        random_state=42,
        n_jobs=-1
    ))
])

# 7. División de datos estratificada (misma estrategia que en Regresión Logística)
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
)

model.fit(X_train, y_train)

# 8. Predicción y Métricas en VALIDACIÓN
print("=" * 50)
print("Métricas en VALIDACIÓN:")
print("=" * 50)
y_val_pred = model.predict(X_val)
y_val_proba = model.predict_proba(X_val)[:, 1]

print(f"Accuracy: {accuracy_score(y_val, y_val_pred):.4f}")
print(f"F1-Score: {f1_score(y_val, y_val_pred):.4f}")
print(f"ROC AUC:  {roc_auc_score(y_val, y_val_proba):.4f}")
print("\nReporte de Clasificación:\n",
      classification_report(y_val, y_val_pred, target_names=['Reprobado', 'Aprobado']))

# 9. Evaluación final en TEST
print("=" * 50)
print("Métricas en TEST (evaluación final):")
print("=" * 50)
y_test_pred = model.predict(X_test)
y_test_proba = model.predict_proba(X_test)[:, 1]

print(f"Accuracy: {accuracy_score(y_test, y_test_pred):.4f}")
print(f"F1-Score: {f1_score(y_test, y_test_pred):.4f}")
print(f"ROC AUC:  {roc_auc_score(y_test, y_test_proba):.4f}")
print("\nReporte de Clasificación:\n",
      classification_report(y_test, y_test_pred, target_names=['Reprobado', 'Aprobado']))

# 10. Importancia de características
print("=" * 50)
print("Importancia de Características (Top 10):")
print("=" * 50)

rf_model = model.named_steps['classifier']
ohe = model.named_steps['preprocessor'].named_transformers_['cat']
cat_feature_names = ohe.get_feature_names_out(columnas_categoricas).tolist()
feature_names = columnas_numericas + cat_feature_names

importancias = pd.Series(rf_model.feature_importances_, index=feature_names)
importancias_sorted = importancias.sort_values(ascending=False)

for feat, imp in importancias_sorted.head(10).items():
    print(f"  {feat:<35} {imp:.4f}")

# 11. (Opcional) Búsqueda de hiperparámetros con GridSearchCV
print("\n" + "=" * 50)
print("Optimización de Hiperparámetros (GridSearchCV):")
print("=" * 50)

param_grid = {
    'classifier__n_estimators': [200, 300],
    'classifier__max_depth': [None, 10, 20],
    'classifier__min_samples_leaf': [1, 2, 4],
    'classifier__max_features': ['sqrt', 'log2']
}

grid_search = GridSearchCV(
    model,
    param_grid,
    cv=5,
    scoring='roc_auc',
    n_jobs=-1,
    verbose=1
)

grid_search.fit(X_train, y_train)

print(f"\nMejores hiperparámetros: {grid_search.best_params_}")
print(f"Mejor ROC AUC (CV):     {grid_search.best_score_:.4f}")

# Evaluación del mejor modelo en TEST
best_model = grid_search.best_estimator_
y_test_pred_best = best_model.predict(X_test)
y_test_proba_best = best_model.predict_proba(X_test)[:, 1]

print("\n--- Métricas del Mejor Modelo en TEST ---")
print(f"Accuracy: {accuracy_score(y_test, y_test_pred_best):.4f}")
print(f"F1-Score: {f1_score(y_test, y_test_pred_best):.4f}")
print(f"ROC AUC:  {roc_auc_score(y_test, y_test_proba_best):.4f}")
print("\nReporte de Clasificación:\n",
      classification_report(y_test, y_test_pred_best, target_names=['Reprobado', 'Aprobado']))