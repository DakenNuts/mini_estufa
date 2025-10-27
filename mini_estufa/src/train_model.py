import pandas as pd
import lightgbm as lgb
from sklearn.metrics import mean_squared_error
import math
import joblib
import os

models_dir = "models"
os.makedirs(models_dir, exist_ok=True)
model_path = os.path.join(models_dir, "modelo_estufa.pkl")

# Carrega os dados
X_train = pd.read_csv('X_train.csv')
X_test = pd.read_csv('X_test.csv')
y_train = pd.read_csv('y_train.csv').values.ravel()
y_test = pd.read_csv('y_test.csv').values.ravel()

# Cria modelo LightGBM
model = lgb.LGBMRegressor(
    objective='regression',
    learning_rate=0.1,
    n_estimators=500,
    max_depth=5
)

# Treina modelo
model.fit(X_train, y_train)

# Avalia
y_pred = model.predict(X_test)
rmse = math.sqrt(mean_squared_error(y_test, y_pred))
print(f"RMSE no teste: {rmse:.2f} °C")

# Salva modelo
joblib.dump(model, model_path)
print(f"Modelo salvo em '{model_path}'")
