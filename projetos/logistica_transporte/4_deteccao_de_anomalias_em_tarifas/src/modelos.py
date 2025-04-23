import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# Bibliotecas para modelagem e avaliação
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, f1_score
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.neighbors import LocalOutlierFactor

# Para o autoencoder
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.callbacks import EarlyStopping

# Carrega os dados
data_path = r"D:\Github\data-science\projetos\logistica_transporte\4_deteccao_de_anomalias_em_tarifas\data\processed\processado.parquet"
df = pd.read_parquet(data_path)

# Seleção de features numéricas (exceto o rótulo 'anomaly')
numeric_features = df.select_dtypes(include=[np.number]).columns.tolist()
if 'anomaly' in numeric_features:
    numeric_features.remove('anomaly')

X = df[numeric_features].copy()
y = df['anomaly'].copy()

# Verifique quais são os rótulos presentes em y (assumindo que sejam -1 e 1)
print("Rótulos presentes no dataset:", np.unique(y))

# Divisão treino/teste (para dados desbalanceados, pode ser interessante usar stratify)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# Normalização dos dados
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

##########################################
# Modelo 1: Isolation Forest (Unsupervised)
##########################################
# IsolationForest já retorna: 1 (inlier) e -1 (outlier)
iso_model = IsolationForest(contamination='auto', random_state=42)
iso_model.fit(X_train_scaled)
y_pred_iso = iso_model.predict(X_test_scaled)
# Mantém a codificação: -1 para anomalia, 1 para normal
print("Isolation Forest Classification Report:")
print(classification_report(y_test, y_pred_iso))

##########################################
# Modelo 2: Local Outlier Factor (Unsupervised)
##########################################
# LOF também retorna: 1 para inlier e -1 para outlier
lof_model = LocalOutlierFactor(n_neighbors=20, contamination='auto')
# Para LOF, aplicamos fit_predict diretamente no conjunto de teste
y_pred_lof = lof_model.fit_predict(X_test_scaled)
print("Local Outlier Factor Classification Report:")
print(classification_report(y_test, y_pred_lof))

##########################################
# Modelo 3: Random Forest (Supervised)
##########################################
# O RandomForest aprende a codificação original (-1 e 1)
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train_scaled, y_train)
y_pred_rf = rf_model.predict(X_test_scaled)
print("Random Forest Classification Report:")
print(classification_report(y_test, y_pred_rf))

##########################################
# Modelo 4: Autoencoder (Unsupervised)
##########################################
# Construindo um autoencoder simples
input_dim = X_train_scaled.shape[1]
encoding_dim = max(1, int(input_dim / 2))  # ajuste conforme o caso

input_layer = Input(shape=(input_dim,))
encoder = Dense(encoding_dim, activation="relu")(input_layer)
decoder = Dense(input_dim, activation="linear")(encoder)

autoencoder = Model(inputs=input_layer, outputs=decoder)
autoencoder.compile(optimizer='adam', loss='mse')

# EarlyStopping para evitar overfitting
early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
autoencoder.fit(
    X_train_scaled, X_train_scaled,
    epochs=100,
    batch_size=32,
    validation_split=0.2,
    verbose=0,
    callbacks=[early_stop]
)

# Calcula o erro de reconstrução no conjunto de teste
reconstructions = autoencoder.predict(X_test_scaled)
mse = np.mean(np.power(X_test_scaled - reconstructions, 2), axis=1)

# Definindo um threshold com base no percentil 95 do erro de reconstrução no conjunto de treino
train_reconstructions = autoencoder.predict(X_train_scaled)
train_mse = np.mean(np.power(X_train_scaled - train_reconstructions, 2), axis=1)
threshold = np.percentile(train_mse, 95)

# Se o erro de reconstrução for maior que o threshold, classifica como anomalia (1), caso contrário normal (0)
y_pred_ae = (mse > threshold).astype(int)
# Mapeia: 1 (anomalia) -> -1 e 0 (normal) -> 1, para manter a mesma codificação
y_pred_ae = np.where(y_pred_ae == 1, -1, 1)

print("Autoencoder Classification Report:")
print(classification_report(y_test, y_pred_ae))

##########################################
# Comparação dos Modelos com base na métrica F1 (anomalías como classe positiva: -1)
##########################################
f1_iso = f1_score(y_test, y_pred_iso, pos_label=-1)
f1_lof = f1_score(y_test, y_pred_lof, pos_label=-1)
f1_rf  = f1_score(y_test, y_pred_rf, pos_label=-1)
f1_ae  = f1_score(y_test, y_pred_ae, pos_label=-1)

scores = {
    'Isolation Forest': f1_iso,
    'Local Outlier Factor': f1_lof,
    'Random Forest': f1_rf,
    'Autoencoder': f1_ae
}

best_model = max(scores, key=scores.get)
print(f"\nMelhor modelo: {best_model} com F1-score: {scores[best_model]:.4f}")
