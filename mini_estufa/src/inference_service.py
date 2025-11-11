import joblib
import numpy as np
import yaml
import time
import os
from mqtt_client import MqttClient

# Caminho do modelo salvo
MODEL_FILE = "models/modelo_estufa.pkl"

def load(cfg_path=os.path.join(os.path.dirname(__file__), "..", "config.yaml")):
    """Carrega configuração YAML e o modelo treinado."""
    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)
    model = joblib.load(MODEL_FILE)

    feature_cols = [
        "greenhous_temperature_celsius",
        "greenhouse_humidity_percentage",
        "greenhouse_illuminance_lux"
    ]
    return cfg, model, feature_cols

def sensor_to_features(sensor_json, feature_cols):
    """Converte JSON de sensores para vetor NumPy de features."""
    x = []
    for f in feature_cols:
        x.append(float(sensor_json.get(f, 0.0)))
    return np.array(x).reshape(1, -1)

def main():
    cfg, model, feature_cols = load()

    mqtt = MqttClient(
        broker=cfg["mqtt"]["broker"],
        port=cfg["mqtt"].get("port", 1883),
        topic_sub=cfg["mqtt"]["topic_sub"],
        topic_pub=cfg["mqtt"]["topic_pub"]
    )

    def handler(data):
        """Função chamada sempre que chega nova mensagem MQTT."""
        try:
            umidade_solo = float(data.get("soil_moisture", 0))
            X = sensor_to_features(data, feature_cols)
            pred = model.predict(X)[0]

            # Lógica de controle
            acionar_bomba = pred < 30 or (umidade_solo < 30)
            cmd = {
                "action": "irrigate",
                "value": int(acionar_bomba)
            }

            mqtt.publish(cfg["mqtt"]["topic_pub"], cmd)

            print(f"\nDados recebidos: {data}")
            print(f"Previsão do modelo: {pred:.2f}")
            print(f"Comando enviado: {cmd}")

        except Exception as e:
            print("Erro na inferência:", e)

    mqtt.add_handler(handler)
    mqtt.start()

    print("Serviço de inferência rodando.")
    print(f"Escutando tópico: {cfg['mqtt']['topic_sub']}")
    print("Pressione Ctrl+C para encerrar.\n")

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("Finalizando serviço...")

if __name__ == "__main__":
    main()
