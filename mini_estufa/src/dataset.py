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
    'marcelboonman/greenhouse-sensor-data-10-minute-interval', 
    path=dataset_dir, 
    unzip=True
)

print("Dataset baixado e descompactado em:", dataset_dir)

# Lista arquivos disponíveis
print("\nArquivos disponíveis no dataset:")
for arquivo in os.listdir(dataset_dir):
    print("-", arquivo)

# Carrega CSV principal
nome_arquivo = '20210703_greenhouse_data.csv'
df = pd.read_csv(os.path.join(dataset_dir, nome_arquivo), sep=';', on_bad_lines='skip', engine='python')

colunas_numericas = [
    'greenhous_temperature_celsius',
    'greenhouse_humidity_percentage',
    'greenhouse_illuminance_lux'
]

for col in colunas_numericas:
    df[col] = df[col].astype(str).str.replace(',', '.', regex=False)
    df[col] = pd.to_numeric(df[col], errors='coerce')

df['created'] = pd.to_datetime(df['created'], errors='coerce')
df = df.dropna(subset=colunas_numericas + ['created']).sort_values('created').reset_index(drop=True)

# Cria target (próxima temperatura)
df['target_temp'] = df['greenhous_temperature_celsius'].shift(-1)

features = colunas_numericas
target = 'target_temp'

X = df[features].iloc[:-1]
y = df[target].iloc[:-1]

# treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Salva CSVs
X_train.to_csv('X_train.csv', index=False)
X_test.to_csv('X_test.csv', index=False)
y_train.to_csv('y_train.csv', index=False)
y_test.to_csv('y_test.csv', index=False)

print("\nDados preparados para treino:")
print(f"X_train: {X_train.shape}, X_test: {X_test.shape}")
print(f"y_train: {y_train.shape}, y_test: {y_test.shape}")
