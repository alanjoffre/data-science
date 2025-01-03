import os
import joblib
import pandas as pd
import logging
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np

# Configurações de logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger()

# Diretórios
model_dir = "D:/Github/data-science/projetos/previsao_de_mercado_de_acoes/random_forest/models/"
results_dir = "D:/Github/data-science/projetos/previsao_de_mercado_de_acoes/random_forest/predictions/"
os.makedirs(model_dir, exist_ok=True)
os.makedirs(results_dir, exist_ok=True)

# Função para avaliar o modelo
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    return rmse, mae, r2

# Fixar a seed para reprodutibilidade
np.random.seed(42)

# Carregando e processando os dados
# Definir o período de análise
start_date = '2010-01-01'
end_date = '2025-01-31'

# Carregando os dados
try:
    import yfinance as yf
    df = yf.download('USDBRL=X', start=start_date, end=end_date)
    df.reset_index(inplace=True)
    df.columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Adj_Close', 'Volume']
    df.dropna(subset=['Open', 'High', 'Low', 'Close', 'Adj_Close'], inplace=True)
    df.drop(columns='Volume', inplace=True)
except Exception as e:
    logger.error("Erro ao carregar os dados: %s", e)
    raise

# Divisão dos dados em features e target
features = df.drop(columns=['Date', 'Close'])
target = df['Close']

# Pré-processamento
imputer = SimpleImputer(strategy='mean')
features = imputer.fit_transform(features)

scaler = StandardScaler()
features = scaler.fit_transform(features)

# Dividindo os dados, mantendo os índices
X_train, X_test, y_train, y_test, train_indices, test_indices = train_test_split(
    features, target, df.index, test_size=0.3, random_state=42)

# Configurando e ajustando o modelo Random Forest
logger.info("Treinando modelo Random Forest...")

# Hiperparâmetros para Grid Search
param_grid = {
    'n_estimators': [100, 200],
    'max_features': ['sqrt'],
    'max_depth': [10, None],
    'min_samples_split': [2],
    'min_samples_leaf': [1]
}

grid_search = GridSearchCV(estimator=RandomForestRegressor(random_state=42), param_grid=param_grid, cv=3, scoring='neg_mean_squared_error')
grid_search.fit(X_train, y_train)

# Melhor modelo
best_rf_model = grid_search.best_estimator_
logger.info("Melhores parâmetros do Random Forest: %s", grid_search.best_params_)

# Avaliando o modelo
rf_rmse, rf_mae, rf_r2 = evaluate_model(best_rf_model, X_test, y_test)
logger.info('Random Forest - RMSE: %.4f, MAE: %.4f, R²: %.4f', rf_rmse, rf_mae, rf_r2)

# Salvando o modelo
model_filename = os.path.join(model_dir, 'best_random_forest_model.joblib')
joblib.dump(best_rf_model, model_filename)
logger.info("Modelo salvo em: %s", model_filename)

# Salvando as previsões com datas
test_dates = df.iloc[test_indices]['Date']  # Pegando as datas correspondentes ao conjunto de teste
predictions = best_rf_model.predict(X_test)

# Criando o DataFrame com previsões e datas
predictions_df = pd.DataFrame({
    'Data': test_dates.values,
    'Real': y_test.values,
    'Previsto': predictions
})

# Ordenando o DataFrame por data em ordem crescente
predictions_df.sort_values(by='Data', inplace=True)

# Salvando o DataFrame ordenado
predictions_df.to_csv(os.path.join(results_dir, 'random_forest_predictions_com_datas.csv'), index=False)
logger.info("Previsões salvas em: %s", os.path.join(results_dir, 'random_forest_predictions_com_datas.csv'))

