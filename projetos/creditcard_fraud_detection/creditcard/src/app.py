# Inicie o Zookeeper
# Acesse: cd C:\kafka_2.13-3.9.0 pelo powershell
# Execute: .\bin\windows\zookeeper-server-start.bat .\config\zookeeper.properties

# Inicie o Kafka
# Acesse: cd C:\kafka_2.13-3.9.0 pelo powershell
# Execute: .\bin\windows\kafka-server-start.bat .\config\server.properties
 
# Crie um tópico no Kafka
# Acesse: cd C:\kafka_2.13-3.9.0 pelo powershell
# .\bin\windows\kafka-topics.bat --create --topic customer_data --bootstrap-server localhost:9092 --partitions 1 --replication-factor 1

# Inicie o Kafka Producer
# Acesse: cd C:\kafka_2.13-3.9.0 pelo powershell
# .\bin\windows\kafka-console-producer.bat --topic customer_data --bootstrap-server localhost:9092

from flask import Flask, request, jsonify
import joblib
import pandas as pd
import traceback

app = Flask(__name__)

# Carregar o modelo e o threshold
model_data = joblib.load(r'D:\Github\data-science\projetos\creditcard_fraud_detection\creditcard\models\best_rf_model_with_threshold.pkl')
best_rf_model = model_data['model']
best_threshold = model_data['threshold']

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json(force=True)
        input_df = pd.DataFrame(data)
        
        # Preprocessar os dados de entrada
        X = input_df.drop(columns=['Class'], errors='ignore')  # Se houver uma coluna 'Class'
        
        # Fazer previsões
        probabilities = best_rf_model.predict_proba(X)[:, 1]
        predictions = (probabilities >= best_threshold).astype(int)
        
        # Converter para tipos compatíveis com JSON
        output = {
            'predictions': list(map(int, predictions)),
            'probabilities': list(map(float, probabilities))
        }
        
        # Retornar as previsões
        return jsonify(output)
    except Exception as e:
        # Registrar o erro no console
        print(f"Erro ao processar a requisição: {e}")
        print(traceback.format_exc())
        return jsonify({'error': 'Erro ao processar a requisição'}), 500

if __name__ == '__main__':
    app.run(debug=True)
