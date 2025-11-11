from kaggle.api.kaggle_api_extended import KaggleApi
import os
import pandas as pd
from sklearn.model_selection import train_test_split

# Inicializa a API do Kaggle
api = KaggleApi()
api.authenticate()

# Pasta onde salva o dataset
dataset_dir = os.path.join(os.getcwd(), 'datasets')
os.makedirs(dataset_dir, exist_ok=True)

# Download e descompacta o dataset
api.dataset_download_files(
    'ziya07/plant-health-data',
    path=dataset_dir,
    unzip=True
)

print("Dataset baixado e descompactado em:", dataset_dir)

# Lista arquivos disponíveis
print("\nArquivos disponíveis no dataset:")
for arquivo in os.listdir(dataset_dir):
    print("-", arquivo)

nome_arquivo = 'Plant_Health_Data.csv'
caminho_arquivo = os.path.join(dataset_dir, nome_arquivo)
if not os.path.exists(caminho_arquivo):
    raise FileNotFoundError(f"Arquivo '{nome_arquivo}' não encontrado em {dataset_dir}.")

df = pd.read_csv(caminho_arquivo)

print("\nColunas encontradas:", list(df.columns))

# Identifica colunas principais
col_temp = [c for c in df.columns if 'temp' in c.lower()][0]
col_hum_ar = [c for c in df.columns if 'humidity' in c.lower() and 'soil' not in c.lower()][0]
col_light = [c for c in df.columns if 'light' in c.lower() or 'illum' in c.lower()][0]
col_soil = [c for c in df.columns if ('soil' in c.lower() and 'humidity' in c.lower()) or 'moisture' in c.lower()][0]
col_time = [c for c in df.columns if 'time' in c.lower() or 'date' in c.lower()][0]
col_plant = [c for c in df.columns if 'plant' in c.lower()][0]

colunas_numericas = [col_temp, col_hum_ar, col_light, col_soil]

print("\nColunas usadas:")
for c in colunas_numericas:
    print("-", c)

# Conversões
df[col_time] = pd.to_datetime(df[col_time], errors='coerce')
for col in colunas_numericas:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Remove linhas inválidas
df = df.dropna(subset=colunas_numericas + [col_time, col_plant]).reset_index(drop=True)

# Ordena por planta e tempo
df = df.sort_values(by=[col_plant, col_time]).reset_index(drop=True)

# Cria o target
df['target_soil_moisture'] = df.groupby(col_plant)[col_soil].shift(-1)

df = df.dropna(subset=['target_soil_moisture']).reset_index(drop=True)

features = colunas_numericas
target = 'target_soil_moisture'

X = df[features]
y = df[target]

# Divide treino e teste (sem embaralhar, pois é temporal)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Salva CSVs
X_train.to_csv('X_train.csv', index=False)
X_test.to_csv('X_test.csv', index=False)
y_train.to_csv('y_train.csv', index=False)
y_test.to_csv('y_test.csv', index=False)

print("\nDados preparados para treino:")
print(f"Entradas (features): {features}")
print(f"Saída (target): {target}")
print(f"X_train: {X_train.shape}, X_test: {X_test.shape}")
print(f"y_train: {y_train.shape}, y_test: {y_test.shape}")

print("\nExemplo de dados de treino:")
print(X_train.head())
print("\nEstatísticas descritivas:")
print(X_train.describe())
