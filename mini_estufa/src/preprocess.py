import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import yaml

def load_config(path="config.yaml"):
    import yaml
    with open(path, "r") as f:
        return yaml.safe_load(f)

def load_dataset(path):
    df = pd.read_csv(path)
    return df

def basic_clean(df):
    # Ajuste conforme colunas do dataset real
    # Ex.: renomear, converter datas, tratar nulos
    df = df.copy()
    # Se houver coluna timestamp
    for col in ["timestamp","time","date"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col])
            break
    # Remover duplicatas
    df = df.drop_duplicates()
    # Exemplo de preenchimento de nulos: interpolação ou median
    df = df.fillna(method="ffill").fillna(method="bfill")
    return df

def feature_engineering(df):
    # Exemplo: extrair hora, dia, médias móveis
    df = df.copy()
    if "timestamp" in df.columns:
        df["hour"] = df["timestamp"].dt.hour
        df["day"] = df["timestamp"].dt.dayofweek
    # Ex: se houver coluna 'lux' ou 'light'
    for c in ["lux","light","luminosity"]:
        if c in df.columns:
            df["light_norm"] = df[c]
            break
    return df

def build_label(df, soil_col="soil_moisture", thresh=30):
    # Cria target binário: 1 = precisa regar, 0 = não precisa
    df = df.copy()
    if soil_col not in df.columns:
        raise ValueError(f"Coluna de umidade do solo '{soil_col}' não encontrada.")
    df["target_irrigation"] = (df[soil_col] < thresh).astype(int)
    return df

def prepare_data(df, feature_cols, target_col="target_irrigation", test_size=0.2, random_state=42):
    X = df[feature_cols].values
    y = df[target_col].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)
    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    return X_train, X_test, y_train, y_test, scaler

if __name__ == "__main__":
    cfg = load_config()
    df = load_dataset(cfg["data"]["path"])
    df = basic_clean(df)
    df = feature_engineering(df)
    df = build_label(df, soil_col=cfg["data"]["soil_col"], thresh=cfg["data"]["soil_thresh"])
    # Automatic feature selection: choose numeric columns excluding target/timestamp
    numerics = df.select_dtypes(include=[np.number]).columns.tolist()
    exclude = [cfg["data"]["soil_col"], "target_irrigation"]
    feature_cols = [c for c in numerics if c not in exclude]
    X_train, X_test, y_train, y_test, scaler = prepare_data(df, feature_cols, target_col="target_irrigation")
    # Save prepared arrays and scaler
    import joblib
    joblib.dump({"X_train": X_train, "X_test": X_test, "y_train": y_train, "y_test": y_test, "feature_cols": feature_cols, "scaler": scaler}, "data/prepared.joblib")
    print("Pré-processamento finalizado. Arquivo salvo: data/prepared.joblib")
