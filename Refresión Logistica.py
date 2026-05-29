import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
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
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), columnas_numericas),
        ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), columnas_categoricas)
    ])

# 6. Construir el modelo
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression( solver = 'saga', l1_ratio = 1, class_weight= 'balanced', random_state=42, max_iter=1000))
])

# 7. División de datos estratificada
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)

model.fit(X_train, y_train)

# 8. Predicción y Métricas
print("=" * 50)
print("Métricas en VALIDACIÓN:")
print("=" * 50)
y_val_pred = model.predict(X_val)
y_val_proba = model.predict_proba(X_val)[:, 1]

print(f"Accuracy: {accuracy_score(y_val, y_val_pred):.4f}")
print(f"F1-Score: {f1_score(y_val, y_val_pred):.4f}")
print(f"ROC AUC:  {roc_auc_score(y_val, y_val_proba):.4f}")
print("\nReporte de Clasificación:\n", classification_report(y_val, y_val_pred, target_names=['Reprobado', 'Aprobado']))

# 9. Evaluación final en conjunto de TEST
print("=" * 50)
print("Métricas en TEST (evaluación final):")
print("=" * 50)
y_test_pred = model.predict(X_test)
y_test_proba = model.predict_proba(X_test)[:, 1]

print(f"Accuracy: {accuracy_score(y_test, y_test_pred):.4f}")
print(f"F1-Score: {f1_score(y_test, y_test_pred):.4f}")
print(f"ROC AUC:  {roc_auc_score(y_test, y_test_proba):.4f}")
print("\nReporte de Clasificación:\n",classification_report(y_test, y_test_pred, target_names=['Reprobado', 'Aprobado']))