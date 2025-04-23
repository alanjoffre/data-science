import joblib
import pandas as pd

# Carregar o modelo
model_path = r"D:/Github/data-science/projetos/daily-bike-shares/src/models/xgb_model.joblib"
model = joblib.load(model_path)

# Função para processar os dados de entrada
def preprocess_input(input_data):
    # Criar DataFrame a partir dos dados de entrada
    input_df = pd.DataFrame(input_data)

    # Realizar as transformações necessárias
    input_df['temp_hum'] = input_df['temp'] * input_df['hum']
    input_df['temp_squared'] = input_df['temp'] ** 2
    input_df['weekday_weekend'] = input_df['workingday'].apply(lambda x: 1 if x == 0 else 0)

    # Codificação one-hot para variáveis categóricas
    input_df = pd.get_dummies(input_df, columns=['season', 'weathersit', 'mnth', 'weekday'], drop_first=True)

    # Garantir que todas as colunas que o modelo espera estão presentes
    model_feature_names = model.get_booster().feature_names
    input_df = input_df.reindex(columns=model_feature_names, fill_value=0)

    return input_df

# Função para fazer previsões
def make_prediction(input_data):
    # Processar os dados de entrada
    input_df = preprocess_input(input_data)

    # Fazer a previsão
    prediction = model.predict(input_df)
    return prediction

# Exemplo de uso
input_data = {
    'temp': [0.5],
    'hum': [0.3],
    'windspeed': [0.1],
    'season': [1],  # 1- Inverno, 2- Primavera, 3- Verão, 4- Outono
    'weathersit': [1],  # 1- Limpo, 2- Nublado, 3- Chuva, 4- Neve
    'mnth': [4],  # 1- Janeiro, 2- Fevereiro, ..., 12- Dezembro
    'weekday': [5],  # 0- Domingo, 1- Segunda, ..., 6- Sábado
    'workingday': [1]  # 1 se dia útil, 0 caso contrário
}

pred = make_prediction(input_data)
print(f"Previsão de aluguel de bicicletas: {pred[0]}")

