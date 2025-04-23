# Importar as bibliotecas
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
import os
import optuna
import warnings

# Suprimir avisos de warnings
warnings.simplefilter("ignore")

# Função para avaliar o modelo
def avaliar_modelo(model, X_test, y_test):
    y_pred = model.predict(X_test)
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    return rmse, mae, r2

# Carregar o dataset
dataset_path = "D:/Github/data-science/projetos/logistica_transporte/1_previsao_gorjeta_regressao/data/processed/dataset_processado.parquet"
df = pd.read_parquet(dataset_path)

# Selecionar apenas as colunas numéricas
X = df.select_dtypes(include=[np.number]).drop(columns=['gorjeta'])
y = df['gorjeta']

# Dividir o dataset em conjunto de treinamento e teste (70/30)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Definir a função objetivo para o Optuna
def objective(trial):
    param = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 500),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'min_samples_split': trial.suggest_int('min_samples_split', 2, 20)
    }
    model = GradientBoostingRegressor(**param, random_state=42)
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(model, X_train, y_train, cv=kf, scoring='neg_mean_squared_error')
    rmse = np.sqrt(-scores.mean())
    return rmse

# Realizar a busca de hiperparâmetros com Optuna
study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=2)  # Sugerido para ao menos 2 trials

# Aplicar os melhores hiperparâmetros
best_params = study.best_params
best_model = GradientBoostingRegressor(**best_params, random_state=42)
best_model.fit(X_train, y_train)

# Avaliar o modelo nos dados de treino
train_rmse, train_mae, train_r2 = avaliar_modelo(best_model, X_train, y_train)
print(f"Desempenho nos dados de treino - RMSE: {train_rmse}, MAE: {train_mae}, R²: {train_r2}")

# Avaliar o modelo nos dados de teste
test_rmse, test_mae, test_r2 = avaliar_modelo(best_model, X_test, y_test)
print(f"Desempenho nos dados de teste - RMSE: {test_rmse}, MAE: {test_mae}, R²: {test_r2}")

# Salvar o modelo
model_save_path = "D:/Github/data-science/projetos/logistica_transporte/1_previsao_gorjeta_regressao/models/gradient_boosting_model.pkl"
os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
joblib.dump(best_model, model_save_path)

print("Modelo salvo com sucesso.")
