import joblib
import pandas as pd
from datetime import datetime
import os

models_dir = "models"
os.makedirs(models_dir, exist_ok=True)
model_path = os.path.join(models_dir, "modelo_estufa.pkl")

historico_dir = "historico"
os.makedirs(historico_dir, exist_ok=True)
historico_csv = os.path.join(historico_dir, "historico_estufa.csv")

features = [
    "greenhous_temperature_celsius",
    "greenhouse_humidity_percentage",
    "greenhouse_illuminance_lux"
]

# Carrega modelo
model = joblib.load(model_path)

# Simulação de leitura dos sensores
temp = 25.3
umidade = 55.0
luminosidade = 300

# Cria dataframe
dados = pd.DataFrame([{
    "greenhous_temperature_celsius": temp,
    "greenhouse_humidity_percentage": umidade,
    "greenhouse_illuminance_lux": luminosidade,
    "created": datetime.now()
}])

# Predição
y_pred = model.predict(dados[features])
dados["predicted_temperature"] = y_pred

# Mostra resultado
print(f"Data/Hora: {dados['created'][0]}")
print(f"Temperatura atual: {temp} °C")
print(f"Previsão temperatura: {y_pred[0]:.2f} °C")

# Salva histórico
if not os.path.exists(historico_csv):
    dados.to_csv(historico_csv, index=False)
else:
    dados.to_csv(historico_csv, mode="a", header=False, index=False)

print(f"Histórico salvo em {historico_csv}")
