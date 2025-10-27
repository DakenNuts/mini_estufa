import paho.mqtt.client as mqtt
import json
import time

class MqttClient:
    def __init__(self, broker, port=1883, topic_sub="estufa/sensores", topic_pub="estufa/atuadores"):
        self.client = mqtt.Client()
        self.client.on_connect = self.on_connect
        self.client.on_message = self.on_message
        self.broker = broker
        self.port = port
        self.topic_sub = topic_sub
        self.topic_pub = topic_pub
        self.handlers = []

    def on_connect(self, client, userdata, flags, rc):
        print("Conectado ao broker MQTT, código:", rc)
        client.subscribe(self.topic_sub)

    def on_message(self, client, userdata, msg):
        payload = msg.payload.decode()
        try:
            data = json.loads(payload)
        except:
            data = payload
        for h in self.handlers:
            h(data)

    def add_handler(self, fn):
        self.handlers.append(fn)

    def start(self):
        self.client.connect(self.broker, self.port, 60)
        self.client.loop_start()

    def publish(self, topic, message):
        self.client.publish(topic, json.dumps(message))
