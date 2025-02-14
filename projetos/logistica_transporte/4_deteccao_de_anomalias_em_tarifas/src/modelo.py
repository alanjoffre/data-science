import os
import time
import pandas as pd
import numpy as np
import warnings
import joblib
import json
import yaml
import random

warnings.filterwarnings('ignore')

# Definir sementes para reprodutibilidade
SEED = 42
np.random.seed(SEED)
random.seed(SEED)

# Importações para modelagem e avaliação
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, f1_score
from sklearn.ensemble import RandomForestClassifier

# Inicia a medição do tempo
start_time = time.time()

# ----------------------------
# Carrega o arquivo de configuração (config.yaml)
# ----------------------------
config_path = r"D:\Github\data-science\projetos\logistica_transporte\4_deteccao_de_anomalias_em_tarifas\config\config.yaml"
with open(config_path, "r") as file:
    config = yaml.safe_load(file)

# Cria diretórios para salvar as métricas e preprocessadores, se não existirem
os.makedirs(config["metrics"]["directory"], exist_ok=True)
os.makedirs(config["models"]["directory"], exist_ok=True)
os.makedirs(os.path.dirname(config["preprocessors"]["path"]), exist_ok=True)

# ----------------------------
# Carrega os dados
# ----------------------------
data_path = config["data"]["processed_parquet"]
df = pd.read_parquet(data_path)

# Seleção de features numéricas (exceto o rótulo 'anomaly')
numeric_features = df.select_dtypes(include=[np.number]).columns.tolist()
if 'anomaly' in numeric_features:
    numeric_features.remove('anomaly')

X = df[numeric_features].copy()
y = df['anomaly'].copy()

print("Rótulos presentes no dataset:", np.unique(y))

# ----------------------------
# Divisão Treino/Teste (com stratify)
# ----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=SEED
)

# ----------------------------
# Pré-processamento: Normalização com StandardScaler
# ----------------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Define o preprocessor como um dicionário contendo as features e o scaler
preprocessor = {"features": numeric_features, "scaler": scaler}

# ----------------------------
# 1. Modelo Random Forest Padrão
# ----------------------------
rf_default = RandomForestClassifier(n_estimators=100, random_state=SEED)
rf_default.fit(X_train_scaled, y_train)
y_pred_default = rf_default.predict(X_test_scaled)

print("----- Random Forest Padrão -----")
print(classification_report(y_test, y_pred_default))
f1_default = f1_score(y_test, y_pred_default, pos_label=-1)
print(f"F1-score (com threshold padrão 0.5): {f1_default:.4f}\n")

# ----------------------------
# 2. Busca de Hiperparâmetros (GridSearchCV)
# ----------------------------
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10]
}
grid_search = GridSearchCV(RandomForestClassifier(random_state=SEED),
                           param_grid,
                           cv=5,
                           scoring='f1',  # Otimizando o F1-score (classe -1)
                           n_jobs=-1)
grid_search.fit(X_train_scaled, y_train)
print("----- GridSearchCV -----")
print("Melhores parâmetros iniciais:", grid_search.best_params_)

best_rf = grid_search.best_estimator_
y_pred_best = best_rf.predict(X_test_scaled)
print("Relatório de Classificação (GridSearchCV):")
print(classification_report(y_test, y_pred_best))
f1_best = f1_score(y_test, y_pred_best, pos_label=-1)
print(f"F1-score (GridSearchCV): {f1_best:.4f}\n")

# ----------------------------
# 3. Refinamento dos Hiperparâmetros (GridSearchCV Refinado)
# ----------------------------
refined_param_grid = {
    'n_estimators': [max(50, grid_search.best_params_['n_estimators'] - 50),
                     grid_search.best_params_['n_estimators'],
                     grid_search.best_params_['n_estimators'] + 50],
    'max_depth': ([grid_search.best_params_['max_depth'] - 5,
                   grid_search.best_params_['max_depth'],
                   grid_search.best_params_['max_depth'] + 5]
                  if grid_search.best_params_['max_depth'] is not None
                  else [None]),
    'min_samples_split': [grid_search.best_params_['min_samples_split'],
                          grid_search.best_params_['min_samples_split'] + 2]
}

refined_search = GridSearchCV(RandomForestClassifier(random_state=SEED),
                              refined_param_grid,
                              cv=5,
                              scoring='f1',
                              n_jobs=-1)
refined_search.fit(X_train_scaled, y_train)
print("----- Refined GridSearchCV -----")
print("Melhores parâmetros refinados:", refined_search.best_params_)

refined_rf = refined_search.best_estimator_
y_pred_refined = refined_rf.predict(X_test_scaled)
print("Relatório de Classificação (Refined GridSearchCV):")
print(classification_report(y_test, y_pred_refined))
f1_refined = f1_score(y_test, y_pred_refined, pos_label=-1)
print(f"F1-score (Refined GridSearchCV): {f1_refined:.4f}\n")

# ----------------------------
# 4. Refinamento Fino dos Hiperparâmetros (GridSearchCV Fine)
# ----------------------------
fine_param_grid = {
    'n_estimators': [refined_search.best_params_['n_estimators'] - 20,
                     refined_search.best_params_['n_estimators'] - 10,
                     refined_search.best_params_['n_estimators'],
                     refined_search.best_params_['n_estimators'] + 10,
                     refined_search.best_params_['n_estimators'] + 20],
    'max_depth': ([refined_search.best_params_['max_depth'] - 2,
                   refined_search.best_params_['max_depth'] - 1,
                   refined_search.best_params_['max_depth'],
                   refined_search.best_params_['max_depth'] + 1,
                   refined_search.best_params_['max_depth'] + 2]
                  if refined_search.best_params_['max_depth'] is not None
                  else [None]),
    'min_samples_split': [max(2, refined_search.best_params_['min_samples_split'] - 1),
                          refined_search.best_params_['min_samples_split'],
                          refined_search.best_params_['min_samples_split'] + 1]
}

fine_search = GridSearchCV(RandomForestClassifier(random_state=SEED),
                           fine_param_grid,
                           cv=5,
                           scoring='f1',
                           n_jobs=-1)
fine_search.fit(X_train_scaled, y_train)
print("----- Fine GridSearchCV -----")
print("Melhores parâmetros finos:", fine_search.best_params_)

fine_rf = fine_search.best_estimator_
y_pred_fine = fine_rf.predict(X_test_scaled)
print("Relatório de Classificação (Fine GridSearchCV):")
print(classification_report(y_test, y_pred_fine))
f1_fine = f1_score(y_test, y_pred_fine, pos_label=-1)
print(f"F1-score (Fine GridSearchCV): {f1_fine:.4f}\n")

# ----------------------------
# 5. Refinamento Ultra Fino dos Hiperparâmetros (GridSearchCV Ultra Fine)
# ----------------------------
ultra_fine_param_grid = {
    'n_estimators': [fine_search.best_params_['n_estimators'] - 5,
                     fine_search.best_params_['n_estimators'],
                     fine_search.best_params_['n_estimators'] + 5],
    'max_depth': ([fine_search.best_params_['max_depth'] - 1,
                   fine_search.best_params_['max_depth'],
                   fine_search.best_params_['max_depth'] + 1]
                  if fine_search.best_params_['max_depth'] is not None
                  else [None]),
    'min_samples_split': [fine_search.best_params_['min_samples_split']]
}

ultra_fine_search = GridSearchCV(RandomForestClassifier(random_state=SEED),
                                 ultra_fine_param_grid,
                                 cv=5,
                                 scoring='f1',
                                 n_jobs=-1)
ultra_fine_search.fit(X_train_scaled, y_train)
print("----- Ultra Fine GridSearchCV -----")
print("Melhores parâmetros ultra finos:", ultra_fine_search.best_params_)

ultra_fine_rf = ultra_fine_search.best_estimator_
y_pred_ultra_fine = ultra_fine_rf.predict(X_test_scaled)
print("Relatório de Classificação (Ultra Fine GridSearchCV):")
print(classification_report(y_test, y_pred_ultra_fine))
f1_ultra_fine = f1_score(y_test, y_pred_ultra_fine, pos_label=-1)
print(f"F1-score (Ultra Fine GridSearchCV): {f1_ultra_fine:.4f}\n")

# ----------------------------
# 6. Validação Cruzada
# ----------------------------
y_train_cv = cross_val_predict(ultra_fine_rf, X_train_scaled, y_train, cv=5)
print("----- Validação Cruzada nos Dados de Treino -----")
print(classification_report(y_train, y_train_cv))

y_test_cv = cross_val_predict(ultra_fine_rf, X_test_scaled, y_test, cv=5)
print("----- Validação Cruzada nos Dados de Teste -----")
print(classification_report(y_test, y_test_cv))

# ----------------------------
# Salvando os objetos finais
# ----------------------------
# Salva o scaler
joblib.dump(scaler, config["models"]["scaler"])
print(f"Scaler salvo em: {config['models']['scaler']}")

# Salva o preprocessor
joblib.dump(preprocessor, config["preprocessors"]["path"])
print(f"Preprocessor salvo em: {config['preprocessors']['path']}")

# Salva o modelo final ultra fino
joblib.dump(ultra_fine_rf, config["models"]["final_model"])
print(f"Modelo final salvo em: {config['models']['final_model']}")

# Salva as métricas obtidas (exemplo: métricas dos modelos padrão, grid search, refined, fine e ultra fine)
metrics = {
    "f1_default": f1_default,
    "f1_gridsearch": f1_best,
    "f1_refined": f1_refined,
    "f1_fine": f1_fine,
    "f1_ultra_fine": f1_ultra_fine
}

metrics_path = os.path.join(config["metrics"]["directory"], "metrics.json")
with open(metrics_path, "w") as f:
    json.dump(metrics, f, indent=4)
print(f"Métricas salvas em: {metrics_path}")

# ----------------------------
# Tempo Total de Processamento
# ----------------------------
end_time = time.time()
total_time = end_time - start_time

hours = int(total_time // 3600)
minutes = int((total_time % 3600) // 60)
seconds = total_time % 60

print(f"\nTempo total de processamento: {hours:02d}hr {minutes:02d}min {seconds:05.2f}seg")
