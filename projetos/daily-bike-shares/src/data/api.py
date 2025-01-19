from flask import Flask, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)

# Carregar o modelo
model_path = r"D:/Github/data-science/projetos/daily-bike-shares/src/models/xgb_model.joblib"
model = joblib.load(model_path)

@app.route('/predict', methods=['POST'])
def make_prediction():
    input_data = request.json  # Use request.json para decodificar o JSON diretamente
    
    # Criar DataFrame a partir dos dados de entrada
    input_df = pd.DataFrame(input_data)
    
    # Realizar transformações necessárias
    input_df['temp_hum'] = input_df['temp'] * input_df['hum']
    input_df['temp_squared'] = input_df['temp'] ** 2
    input_df['weekday_weekend'] = input_df['workingday'].apply(lambda x: 1 if x == 0 else 0)
    
    # Criar dummies para variáveis categóricas
    input_df = pd.get_dummies(input_df, columns=['season', 'weathersit', 'mnth', 'weekday'], drop_first=True)

    # Obter a lista de colunas que o modelo espera
    model_columns = model.get_booster().feature_names
    
    # Garantir que todas as colunas que o modelo espera estejam presentes no input_df
    input_df = input_df.reindex(columns=model_columns, fill_value=0)

    # Fazer a previsão
    prediction = model.predict(input_df)
    return jsonify(prediction.tolist())  # Retornar a previsão como JSON

if __name__ == '__main__':
    app.run(debug=True)
