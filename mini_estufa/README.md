# Mini Estufa Inteligente

Sistema de previsão de temperatura da estufa utilizando sensores de:
- Temperatura (DHT22)
- Umidade do ar (DHT22)
- Luminosidade (LDR)

## Estrutura do projeto

- `dataset.py` → Baixa, limpa e prepara os dados do Kaggle.
- `train_model.py` → Treina modelo LightGBM para previsão de temperatura futura.
- `predict.py` → Simula leitura dos sensores e faz previsões, salvando histórico com data/hora.
- `inference_service.py` → Serviço de inferência via MQTT para sensores reais (em tempo real).
- `mqtt_client.py` → Cliente MQTT genérico.
- `preprocess.py` → Funções de pré-processamento (opcional, usado se houver outro dataset).

## Dependências

Instalar via `pip`:

```bash
pip install -r requirements.txt
