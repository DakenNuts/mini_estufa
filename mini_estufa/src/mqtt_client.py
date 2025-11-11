import paho.mqtt.client as mqtt
import json
import time

class MqttClient:
    def __init__(self, broker, port=1883, topic_sub="estufa/sensores", topic_pub="estufa/atuadores"):
        self.client = mqtt.Client(protocol=mqtt.MQTTv311)
        self.client.on_connect = self.on_connect
        self.client.on_message = self.on_message

        self.broker = broker
        self.port = port
        self.topic_sub = topic_sub
        self.topic_pub = topic_pub

        self.handlers = []

    def on_connect(self, client, userdata, flags, rc):
        if rc == 0:
            print(f"Conectado ao broker MQTT ({self.broker}:{self.port})")
            client.subscribe(self.topic_sub)
            print(f"Inscrito no tópico: {self.topic_sub}")
        else:
            print(f"Falha na conexão. Código de erro: {rc}")

    def on_message(self, client, userdata, msg):
        payload = msg.payload.decode()
        try:
            data = json.loads(payload)
        except json.JSONDecodeError:
            data = payload
        for h in self.handlers:
            h(data)

    def add_handler(self, fn):
        """Adiciona uma função para lidar com mensagens recebidas."""
        self.handlers.append(fn)

    def start(self):
        """Conecta ao broker e mantém o loop ativo em segundo plano."""
        while True:
            try:
                self.client.connect(self.broker, self.port, 60)
                break
            except Exception as e:
                print(f"Erro ao conectar ao broker: {e}. Tentando novamente em 5s...")
                time.sleep(5)
        self.client.loop_start()

    def publish(self, topic, message):
        """Publica uma mensagem no tópico especificado."""
        if isinstance(message, (dict, list)):
            message = json.dumps(message)
        self.client.publish(topic, message)
        print(f"Publicado em '{topic}': {message}")
