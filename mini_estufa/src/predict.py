import joblib
import pandas as pd
from datetime import datetime
import os
import paho.mqtt.client as mqtt

models_dir = "models"
os.makedirs(models_dir, exist_ok=True)
model_path = os.path.join(models_dir, "modelo_estufa.pkl")
features_path = os.path.join(models_dir, "features.joblib")

historico_dir = "historico"
os.makedirs(historico_dir, exist_ok=True)
historico_csv = os.path.join(historico_dir, "historico_estufa.csv")

# Configuração MQTT
BROKER = "test.mosquitto.org"   # <- IP da Raspberry
TOPIC = "estufa/controle"  # <- tópico de publicação
client = mqtt.Client()
client.connect(BROKER, 1883, 60)

model = joblib.load(model_path)
features = joblib.load(features_path)

# Simulação de leitura dos sensores
dados_sensor = {
    "Ambient_Temperature": 27.5,
    "Humidity": 60.0,
    "Light_Intensity": 700,
    "Soil_Moisture": 22.0,
    "created": datetime.now()
}

df = pd.DataFrame([dados_sensor])

# Predição
y_pred = model.predict(df[features])[0]
regar_por_modelo = bool(y_pred)

umidade_solo = df["Soil_Moisture"][0]
regar_por_regra = umidade_solo < 30

regar = regar_por_modelo or regar_por_regra

if regar:
    decisao = "Regar agora"
    comando = "LIGAR_BOMBA"
else:
    decisao = "Não regar"
    comando = "DESLIGAR_BOMBA"

client.publish(TOPIC, comando)
print(f"Comando MQTT enviado: {comando}")

# Exibe dados e decisão
print(f"\nData/Hora: {df['created'][0]}")
print(f"Temperatura: {df['Ambient_Temperature'][0]} °C")
print(f"Umidade do ar: {df['Humidity'][0]} %")
print(f"Luminosidade: {df['Light_Intensity'][0]} lux")
print(f"Umidade do solo: {df['Soil_Moisture'][0]} %")
print(f"Decisão da IA: {decisao}")

# Salva histórico
if not os.path.exists(historico_csv):
    df.assign(prediction=decisao).to_csv(historico_csv, index=False)
else:
    df.assign(prediction=decisao).to_csv(historico_csv, mode="a", header=False, index=False)

print(f"Histórico salvo em {historico_csv}")
