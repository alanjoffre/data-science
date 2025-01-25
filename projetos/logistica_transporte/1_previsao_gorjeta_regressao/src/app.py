import joblib
import pandas as pd
from flask import Flask, request, jsonify, Response
import logging
from prometheus_client import start_http_server, Counter, generate_latest

# Configuração do Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Carregar o pipeline salvo
model_path = "D:/Github/data-science/projetos/logistica_transporte/1_previsao_gorjeta_regressao/models/pipeline_model.pkl"
try:
    model = joblib.load(model_path)
    logging.info("Modelo carregado com sucesso de %s", model_path)
except Exception as e:
    logging.error("Erro ao carregar o modelo: %s", str(e))
    raise

# Inicializar o Flask
app = Flask(__name__)

# Iniciar o servidor HTTP para expor as métricas do Prometheus
start_http_server(8000)

# Inicializar um contador para as requisições
REQUEST_COUNT = Counter('request_count', 'Number of requests')

@app.route('/metrics')
def metrics():
    return Response(generate_latest(), mimetype='text/plain')

@app.route('/')
def home():
    """Rota inicial para verificar se a API está ativa."""
    REQUEST_COUNT.inc()
    return "API de Previsão de Gorjetas está ativa e funcionando!"

@app.route('/predict', methods=['POST'])
def predict():
    """Rota para realizar previsões de gorjetas."""
    logging.info("A rota /predict foi acessada.")
    REQUEST_COUNT.inc()
    try:
        # Obter os dados do corpo da requisição
        data = request.get_json()
        if not data:
            raise ValueError("Nenhum dado fornecido na requisição.")
        
        # Validar se os dados estão no formato esperado
        if not isinstance(data, list):
            raise ValueError("Os dados fornecidos devem ser uma lista de objetos JSON.")

        # Converter os dados em um DataFrame
        df = pd.DataFrame(data)
        logging.info("Dados recebidos: %s", df.to_dict(orient='records'))

        # Verificar se a coluna 'total_da_viagem' existe no DataFrame
        if 'total_da_viagem' not in df.columns:
            raise ValueError("A coluna 'total_da_viagem' é obrigatória nos dados enviados.")

        # Realizar a previsão
        prediction = model.predict(df)
        logging.info("Previsão gerada: %s", prediction.tolist())
        
        # Calcular o valor da gorjeta
        df['gorjeta_prevista'] = prediction
        df['valor_gorjeta'] = df['total_da_viagem'] * df['gorjeta_prevista']

        # Preparar a saída detalhada
        response = {
            'previsao_porcentagem': df['gorjeta_prevista'].tolist(),
            'valor_gorjeta': df['valor_gorjeta'].tolist(),
            'mensagem': (
                f"Para a viagem fornecida, o modelo previu uma gorjeta de "
                f"aproximadamente {round(prediction[0] * 100, 2)}% do valor total da viagem, "
                f"o que equivale a aproximadamente {round(df['valor_gorjeta'].iloc[0], 2)} na moeda utilizada."
            )
        }

        return jsonify(response)

    except ValueError as ve:
        logging.error("Erro de validação: %s", str(ve))
        return jsonify({'error': f"Erro de validação: {str(ve)}"}), 400
    except Exception as e:
        logging.error("Erro ao processar a requisição: %s", str(e))
        return jsonify({'error': f"Erro interno: {str(e)}"}), 500

if __name__ == '__main__':
    # Iniciar o servidor Flask na porta 5000 e garantir que o host seja acessível
    app.run(debug=True, host='0.0.0.0', port=5000)
