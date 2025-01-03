import os
import joblib
import pandas as pd
import logging
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import statsmodels.api as sm
from statsmodels.tsa.arima.model import ARIMA
from arch import arch_model
import numpy as np
import shutil
from tempfile import mkdtemp
import yfinance as yf


# Função para avaliar os modelos
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    return rmse, mae, r2

# Configurações do logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger()

# Diretórios
model_dir = "D:/Github/data-science/projetos/previsao_de_mercado_de_acoes/random_forest/models/"
preprocessors_dir = "D:/Github/data-science/projetos/previsao_de_mercado_de_acoes/random_forest/preprocessors/"
data_dir = "D:/Github/data-science/projetos/previsao_de_mercado_de_acoes/random_forest/data/raw/"
results_dir = "D:/Github/data-science/projetos/previsao_de_mercado_de_acoes/random_forest/predictions/"
figures_dir = "D:/Github/data-science/projetos/previsao_de_mercado_de_acoes/random_forest/reports/figures/"
os.makedirs(model_dir, exist_ok=True)
os.makedirs(preprocessors_dir, exist_ok=True)
os.makedirs(results_dir, exist_ok=True)
os.makedirs(figures_dir, exist_ok=True)

# Definindo o período
start_date = '2010-01-01'
end_date = '2025-01-31'

# Coletando dados de câmbio dólar/real
ticker = 'USDBRL=X'
df = yf.download(ticker, start=start_date, end=end_date)
df.reset_index(inplace=True)
df.columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Adj_Close', 'Volume']

# Limpeza de dados
df.dropna(subset=['Open', 'High', 'Low', 'Close', 'Adj_Close'], inplace=True)
df.drop(columns='Volume', inplace=True)
df = df[(df['Open'] > 0) & (df['High'] > 0) & (df['Low'] > 0) & (df['Close'] > 0) & (df['Adj_Close'] > 0)]

# Garantir que o log seja calculado apenas para valores positivos
if (df['Close'] <= 0).any():
    print("Aviso: Alguns valores em 'Close' são <= 0 e serão ignorados no cálculo do log.")
    df = df[df['Close'] > 0]

# Cálculo do logaritmo
df['Log_Close'] = np.log(df['Close'])

# Divisão dos Dados e Validação Cruzada
features = df.drop(columns=['Date', 'Close'])
target = df['Close']
imputer = SimpleImputer(strategy='mean')
features = imputer.fit_transform(features)

# Definição do scaler (ou carregue o scaler salvo)
scaler = StandardScaler()
features = scaler.fit_transform(features)
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.3, random_state=42)

# Validação Cruzada com Regressão Linear
linear_model = LinearRegression()
cv_scores = cross_val_score(linear_model, X_train, y_train, cv=5, scoring='neg_mean_squared_error')
rmse_scores = np.sqrt(-cv_scores)
logger.info("Validação Cruzada (RMSE) - Regressão Linear: %s", rmse_scores)

# Modelos Estatísticos
ts_data = df.set_index('Date')['Close']

# ARIMA
arima_model = ARIMA(ts_data, order=(5, 1, 0))
arima_model_fit = arima_model.fit()
logger.info(arima_model_fit.summary())

# GARCH
garch_model = arch_model(ts_data, vol='Garch', p=1, q=1)
garch_model_fit = garch_model.fit()
logger.info(garch_model_fit.summary())

# Suavização Exponencial
exp_smooth_model = sm.tsa.ExponentialSmoothing(ts_data, seasonal='add', seasonal_periods=12).fit()
logger.info(exp_smooth_model.summary())

# Modelos de Machine Learning
# Regressão Linear
linear_model.fit(X_train, y_train)
linear_rmse, linear_mae, linear_r2 = evaluate_model(linear_model, X_test, y_test)
logger.info('Regressão Linear: RMSE = %f, MAE = %f, R² = %f', linear_rmse, linear_mae, linear_r2)

# Random Forest
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
rf_rmse, rf_mae, rf_r2 = evaluate_model(rf_model, X_test, y_test)
logger.info('Random Forest: RMSE = %f, MAE = %f, R² = %f', rf_rmse, rf_mae, rf_r2)

# XGBoost
xgb_model = XGBRegressor(n_estimators=100, random_state=42)
xgb_model.fit(X_train, y_train)
xgb_rmse, xgb_mae, xgb_r2 = evaluate_model(xgb_model, X_test, y_test)
logger.info('XGBoost: RMSE = %f, MAE = %f, R² = %f', xgb_rmse, xgb_mae, xgb_r2)

# Redes Neurais Simples
nn_model = Sequential()
nn_model.add(Dense(64, input_dim=X_train.shape[1], activation='relu'))
nn_model.add(Dense(32, activation='relu'))
nn_model.add(Dense(1, activation='linear'))
nn_model.compile(loss='mean_squared_error', optimizer='adam')
nn_model.fit(X_train, y_train, epochs=50, batch_size=10, verbose=1)
nn_rmse, nn_mae, nn_r2 = evaluate_model(nn_model, X_test, y_test)
logger.info('Redes Neurais Simples: RMSE = %f, MAE = %f, R² = %f', nn_rmse, nn_mae, nn_r2)

# LSTM
X_train_lstm = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
X_test_lstm = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

lstm_model = Sequential()
lstm_model.add(LSTM(50, return_sequences=True, input_shape=(X_train_lstm.shape[1], 1)))
lstm_model.add(LSTM(50, return_sequences=False))
lstm_model.add(Dense(1))
lstm_model.compile(optimizer='adam', loss='mean_squared_error')
lstm_model.fit(X_train_lstm, y_train, epochs=50, batch_size=10, verbose=1)

lstm_rmse, lstm_mae, lstm_r2 = evaluate_model(lstm_model, X_test_lstm, y_test)
logger.info('LSTM: RMSE = %f, MAE = %f, R² = %f', lstm_rmse, lstm_mae, lstm_r2)

# Comparando os Resultados e Escolhendo o Melhor Modelo
model_results = {
    "Regressão Linear": {"RMSE": linear_rmse, "MAE": linear_mae, "R²": linear_r2},
    "Random Forest": {"RMSE": rf_rmse, "MAE": rf_mae, "R²": rf_r2},
    "XGBoost": {"RMSE": xgb_rmse, "MAE": xgb_mae, "R²": xgb_r2},
    "Redes Neurais Simples": {"RMSE": nn_rmse, "MAE": nn_mae, "R²": nn_r2},
    "LSTM": {"RMSE": lstm_rmse, "MAE": lstm_mae, "R²": lstm_r2}
}

best_model = min(model_results, key=lambda k: model_results[k]["RMSE"])
logger.info("O melhor modelo é %s com RMSE = %f, MAE = %f, e R² = %f", best_model, model_results[best_model]['RMSE'], model_results[best_model]['MAE'], model_results[best_model]['R²'])

# Salvar os resultados dos modelos
results_df = pd.DataFrame(model_results).transpose()
results_df.to_csv(os.path.join(figures_dir, 'model_results.csv'))

# Configurar backend para evitar problemas com tkinter
import matplotlib
matplotlib.use('Agg')

# Configuração de recursos temporários
temp_folder = mkdtemp()
os.environ['JOBLIB_TEMP_FOLDER'] = temp_folder

# Fixar a seed para reprodutibilidade
np.random.seed(42)

# Definindo o Grid de Hiperparâmetros
param_grid = {
    'n_estimators': [100, 200],
    'max_features': ['sqrt'],
    'max_depth': [10, None],
    'min_samples_split': [2],
    'min_samples_leaf': [1]
}

# Grid Search com validação cruzada
logger.info("Iniciando GridSearchCV...")
grid_search = GridSearchCV(estimator=RandomForestRegressor(), param_grid=param_grid, cv=3)
grid_search.fit(X_train, y_train)
logger.info(f"Melhores parâmetros: {grid_search.best_params_}")

# Salvar o modelo final com melhores parâmetros
best_rf_model = grid_search.best_estimator_
model_filename = os.path.join(model_dir, 'best_random_forest_model.joblib')
joblib.dump(best_rf_model, model_filename)
logger.info(f"Melhor modelo salvo em: {model_filename}")
