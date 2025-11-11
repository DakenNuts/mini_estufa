import joblib
import os
import math
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import lightgbm as lgb

models_dir = "models"
os.makedirs(models_dir, exist_ok=True)
model_path = os.path.join(models_dir, "modelo_estufa.pkl")
features_path = os.path.join(models_dir, "features.joblib")

print("Carregando dados pré-processados...")
data = joblib.load("data/prepared.joblib")
X_train, X_test, y_train, y_test = data["X_train"], data["X_test"], data["y_train"], data["y_test"]
feature_cols = data["feature_cols"]

print("Treinando modelo de classificação (regar/não regar)...")

model = lgb.LGBMClassifier(
    objective="binary",
    learning_rate=0.05,
    n_estimators=500,
    num_leaves=64,
    max_depth=-1
)

model.fit(X_train, y_train)

# Avaliação
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

print(f"\nAcurácia no teste: {acc*100:.2f}%")
print("\nMatriz de confusão:")
print(cm)
print("\nRelatório de classificação:")
print(classification_report(y_test, y_pred))

# Salva modelo
joblib.dump(model, model_path)
joblib.dump(feature_cols, features_path)
print(f"\nModelo salvo em '{model_path}'")
print(f"Features salvas em '{features_path}'")
