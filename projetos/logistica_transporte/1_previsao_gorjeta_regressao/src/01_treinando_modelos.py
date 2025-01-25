import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import randint

# Importar o dataset processado
dataset_path = "D:/Github/data-science/projetos/logistica_transporte/1_previsao_gorjeta_regressao/data/processed/dataset_processado.parquet"
df = pd.read_parquet(dataset_path)

# Obter uma amostra de 10% do dataset
df_sample = df.sample(frac=0.1, random_state=42)

# Selecionar apenas as colunas numéricas para o treinamento
X = df_sample.select_dtypes(include=[np.number]).drop(columns=['gorjeta'])
y = df_sample['gorjeta']

# Dividir o dataset em conjuntos de treinamento (80%) e teste (20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

def avaliar_modelo(model, X_test, y_test):
    y_pred = model.predict(X_test)
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    return rmse, mae, r2

# Dicionário para armazenar as métricas de cada modelo
resultados = {}

# Treinamento e Avaliação da Regressão Linear
print("Treinamento da Regressão Linear")
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)
rmse, mae, r2 = avaliar_modelo(lr_model, X_test, y_test)
resultados["Regressão Linear"] = (rmse, mae, r2)
print(f"Regressão Linear - RMSE: {rmse}, MAE: {mae}, R²: {r2}")

# Treinamento e Avaliação do Random Forest Regressor
print("Treinamento do Random Forest Regressor")
rf_model = RandomForestRegressor(random_state=42)
rf_model.fit(X_train, y_train)
rmse, mae, r2 = avaliar_modelo(rf_model, X_test, y_test)
resultados["Random Forest Regressor"] = (rmse, mae, r2)
print(f"Random Forest Regressor - RMSE: {rmse}, MAE: {mae}, R²: {r2}")

# Ajuste de Hiperparâmetros com Random Search para Random Forest Regressor
print("Ajuste de Hiperparâmetros com Random Search para Random Forest Regressor")
param_dist = {
    'n_estimators': randint(100, 500),
    'max_features': [1.0, 'sqrt', 'log2'],
    'max_depth': randint(4, 20),
    'min_samples_split': randint(2, 10),
    'min_samples_leaf': randint(1, 10),
}
rs_rf_model = RandomizedSearchCV(rf_model, param_distributions=param_dist, n_iter=50, cv=3, random_state=42, n_jobs=-1)
rs_rf_model.fit(X_train, y_train)
best_rf_model = rs_rf_model.best_estimator_
rmse, mae, r2 = avaliar_modelo(best_rf_model, X_test, y_test)
resultados["Random Forest Regressor (Ajustado)"] = (rmse, mae, r2)
print(f"Random Forest Regressor (Ajustado) - RMSE: {rmse}, MAE: {mae}, R²: {r2}")

# Treinamento e Avaliação do Gradient Boosting Regressor
print("Treinamento do Gradient Boosting Regressor")
gb_model = GradientBoostingRegressor(random_state=42)
gb_model.fit(X_train, y_train)
rmse, mae, r2 = avaliar_modelo(gb_model, X_test, y_test)
resultados["Gradient Boosting Regressor"] = (rmse, mae, r2)
print(f"Gradient Boosting Regressor - RMSE: {rmse}, MAE: {mae}, R²: {r2}")

# Ajuste de Hiperparâmetros com Grid Search para Gradient Boosting Regressor
print("Ajuste de Hiperparâmetros com Grid Search para Gradient Boosting Regressor")
param_grid = {
    'n_estimators': [100, 200, 300],
    'learning_rate': [0.01, 0.05, 0.1],
    'max_depth': [3, 4, 5],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
}
gs_gb_model = GridSearchCV(gb_model, param_grid=param_grid, cv=3, n_jobs=-1)
gs_gb_model.fit(X_train, y_train)
best_gb_model = gs_gb_model.best_estimator_
rmse, mae, r2 = avaliar_modelo(best_gb_model, X_test, y_test)
resultados["Gradient Boosting Regressor (Ajustado)"] = (rmse, mae, r2)
print(f"Gradient Boosting Regressor (Ajustado) - RMSE: {rmse}, MAE: {mae}, R²: {r2}")

# Treinamento e Avaliação do KNeighbors Regressor
print("Treinamento do KNeighbors Regressor")
knn_model = KNeighborsRegressor()
knn_model.fit(X_train, y_train)
rmse, mae, r2 = avaliar_modelo(knn_model, X_test, y_test)
resultados["KNeighbors Regressor"] = (rmse, mae, r2)
print(f"KNeighbors Regressor - RMSE: {rmse}, MAE: {mae}, R²: {r2}")

# Treinamento e Avaliação do SVR
print("Treinamento do SVR")
svr_model = SVR()
svr_model.fit(X_train, y_train)
rmse, mae, r2 = avaliar_modelo(svr_model, X_test, y_test)
resultados["SVR"] = (rmse, mae, r2)
print(f"SVR - RMSE: {rmse}, MAE: {mae}, R²: {r2}")

# Treinamento e Avaliação do XGBRegressor
print("Treinamento do XGBRegressor")
xgb_model = XGBRegressor(random_state=42)
xgb_model.fit(X_train, y_train)
rmse, mae, r2 = avaliar_modelo(xgb_model, X_test, y_test)
resultados["XGBRegressor"] = (rmse, mae, r2)
print(f"XGBRegressor - RMSE: {rmse}, MAE: {mae}, R²: {r2}")

# Encontrar o melhor modelo com base no RMSE
melhor_modelo = min(resultados, key=lambda k: resultados[k][0])
print(f"Melhor modelo: {melhor_modelo} com RMSE: {resultados[melhor_modelo][0]}, MAE: {resultados[melhor_modelo][1]}, R²: {resultados[melhor_modelo][2]}")

print("Treinamento e Avaliação dos Modelos Concluídos.")
