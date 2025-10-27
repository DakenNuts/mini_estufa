import joblib
import numpy as np
import yaml
from mqtt_client import MqttClient

cfg = {}
MODEL_FILE = "models/modelo_estufa.pkl"

def load(cfg_path="config.yaml"):
    global cfg
    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)
    model = joblib.load(MODEL_FILE)
    feature_cols = [
        "greenhous_temperature_celsius",
        "greenhouse_humidity_percentage",
        "greenhouse_illuminance_lux"
    ]
    return model, feature_cols

def sensor_to_features(sensor_json, feature_cols):
    x = []
    for f in feature_cols:
        x.append(float(sensor_json.get(f, 0.0)))
    return np.array(x).reshape(1, -1)

def main():
    model, feature_cols = load()
    mqtt = MqttClient(
        broker=cfg["mqtt"]["broker"],
        topic_sub=cfg["mqtt"]["topic_sub"],
        topic_pub=cfg["mqtt"]["topic_pub"]
    )

    def handler(data):
        try:
            X = sensor_to_features(data, feature_cols)
            pred = model.predict(X)[0]
            cmd = {"action": "irrigate", "value": int(pred < 25)}
            mqtt.publish(cfg["mqtt"]["topic_pub"], cmd)
            print("Sensor:", data, "=> pred:", pred, " cmd:", cmd)
        except Exception as e:
            print("Erro na inferência:", e)

    mqtt.add_handler(handler)
    mqtt.start()
    print("Serviço de inferência rodando. Escutando tópico:", cfg["mqtt"]["topic_sub"])
    try:
        while True:
            import time
            time.sleep(1)
    except KeyboardInterrupt:
        print("Finalizando serviço...")

if __name__ == "__main__":
    main()
