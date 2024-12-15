import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Função para carregar e pré-processar os dados
def load_and_preprocess_data(file_path):
    """Carrega e pré-processa os dados do arquivo CSV."""
    data = pd.read_csv(file_path)
    
    # Criar novas variáveis
    data['temp_hum'] = data['temp'] * data['hum']
    data['temp_squared'] = data['temp'] ** 2
    data['weekday_weekend'] = data['workingday'].apply(lambda x: 1 if x == 0 else 0)

    # Codificação one-hot para as variáveis categóricas
    data = pd.get_dummies(data, columns=['season', 'weathersit', 'mnth', 'weekday'], drop_first=True)

    return data

# Função para dividir dados e treinar o modelo
def train_model(data):
    """Divide os dados e treina um modelo XGBoost."""
    X = data.drop(columns=['instant', 'dteday', 'rentals'])  # Features
    y = data['rentals']  # Target

    # Dividir os dados
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Treinamento do modelo
    model = XGBRegressor(objective='reg:squarederror')
    model.fit(X_train, y_train)

    # Previsões e avaliação
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    print(f'MAE: {mae}, RMSE: {rmse}, R²: {r2}')
    
    return model, X_train.columns

# Função para otimizar hiperparâmetros usando GridSearchCV
def optimize_xgboost(X_train, y_train):
    """Otimiza hiperparâmetros usando GridSearchCV."""
    param_grid = {
        'learning_rate': [0.01, 0.05, 0.1],
        'max_depth': [3, 5, 7],
        'n_estimators': [100, 300],
        'subsample': [0.8, 1.0],
    }

    grid_search = GridSearchCV(XGBRegressor(objective='reg:squarederror'), param_grid, cv=3, scoring='neg_mean_squared_error', verbose=1)
    grid_search.fit(X_train, y_train)

    print("Melhores Hiperparâmetros encontrados:", grid_search.best_params_)
    print("Melhor RMSE:", np.sqrt(-grid_search.best_score_))

    return grid_search.best_estimator_

# Função para plotar resultados
def plot_results(y_test, y_pred):
    """Plota resultados de previsões."""
    plt.figure(figsize=(10, 5))
    sns.scatterplot(x=y_test, y=y_pred)
    plt.xlabel("Valores Reais")
    plt.ylabel("Previsões")
    plt.title("Valores Reais vs Previsões")
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
    plt.show()

    # Histogramas de erros
    errors = y_pred - y_test
    plt.figure(figsize=(10, 5))
    sns.histplot(errors, bins=30, kde=True)
    plt.title("Distribuição dos Erros de Previsão")
    plt.xlabel("Erro")
    plt.ylabel("Frequência")
    plt.show()

# Função para pré-processar dados de entrada para previsão
def preprocess_input(input_data, model_feature_names):
    """Pré-processa os dados de entrada para previsão."""
    input_data['temp_hum'] = input_data['temp'] * input_data['hum']
    input_data['temp_squared'] = input_data['temp'] ** 2
    input_data['weekday_weekend'] = input_data['workingday'].apply(lambda x: 1 if x == 0 else 0)

    # Codificação one-hot para as variáveis categóricas
    input_data = pd.get_dummies(input_data, columns=['season', 'weathersit', 'mnth', 'weekday'], drop_first=True)

    # Garantir que todas as colunas que o modelo espera estão presentes, mesmo que ausentes
    input_data = input_data.reindex(columns=model_feature_names, fill_value=0)

    return input_data

# Função para fazer previsões
def make_prediction(input_data):
    """Faz previsões usando o modelo carregado."""
    model_path = r"D:\Github\data-science\projetos\daily-bike-shares\src\models\xgb_model.joblib"
    model = joblib.load(model_path)

    # Preprocessar os dados de entrada
    input_df = preprocess_input(input_data, model.get_booster().feature_names)

    # Prever
    prediction = model.predict(input_df)
    return prediction

# Função principal
def main():
    file_path = r"D:\Github\data-science\projetos\daily-bike-shares\data\raw\daily-bike-share.csv"
    data = load_and_preprocess_data(file_path)

    # Treinamento do modelo
    model, feature_names = train_model(data)

    # Otimização de hiperparâmetros
    best_model = optimize_xgboost(data[feature_names], data['rentals'])

    # Salvar o modelo otimizado
    joblib.dump(best_model, r"D:\Github\data-science\projetos\daily-bike-shares\src\models\xgb_model.joblib")

    # Avaliar o modelo otimizado
    y_test = data['rentals']
    y_pred = best_model.predict(data[feature_names])
    plot_results(y_test, y_pred)

# Exemplo de previsão
def predict_example():
    """Faz uma previsão de exemplo usando dados de entrada."""
    input_data = pd.DataFrame({
        'yr': [0],  # 0 para 2011, 1 para 2012
        'holiday': [0],  # 1 se feriado, 0 caso contrário
        'workingday': [1],  # 1 se dia útil, 0 caso contrário
        'temp': [0.5],  # Temperatura normalizada
        'atemp': [0.5],  # Temperatura ajustada normalizada
        'hum': [0.4],  # Umidade normalizada
        'windspeed': [0.1],  # Velocidade do vento normalizada
        'season': [1],  # 1- Inverno, 2- Primavera, 3- Verão, 4- Outono
        'weathersit': [1],  # 1- Limpo, 2- Nublado, 3- Chuva, 4- Neve
        'mnth': [1],  # 1- Janeiro, 2- Fevereiro, ..., 12- Dezembro
        'weekday': [0],  # 0- Domingo, 1- Segunda, ..., 6- Sábado
    })

    pred = make_prediction(input_data)
    print("Previsão de aluguel:", pred)

if __name__ == "__main__":
    main()
    predict_example()
