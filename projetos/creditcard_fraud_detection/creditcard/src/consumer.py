from kafka import KafkaConsumer
import json
import requests
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import yaml

# Carregar o arquivo de configuração
with open(r'D:\Github\data-science\projetos\creditcard_fraud_detection\creditcard\config\config.yaml', 'r') as file:
    config = yaml.safe_load(file)

# Carregar o modelo salvo e o objeto SMOTE
model_data = joblib.load(config['paths']['models_path'] + 'best_rf_model_with_threshold.pkl')
best_rf_model = model_data['model']
best_threshold = model_data['threshold']

# Função para ajustar o threshold
def ajustar_threshold(probabilities, threshold):
    probabilities = np.array(probabilities)  # Converta a lista para um array NumPy
    return np.where(probabilities >= threshold, 1, 0)

# Inicializar o consumidor Kafka
consumer = KafkaConsumer(
    'transactions',
    bootstrap_servers='localhost:9092',
    value_deserializer=lambda x: json.loads(x.decode('utf-8'))
)

print("Iniciando a ingestão de dados e previsão...")

# Listas para armazenar resultados
true_labels = []
pred_labels = []

for message in consumer:
    transaction = message.value
    
    # Adicione o rótulo verdadeiro à lista
    true_labels.append(transaction['Class'])
    
    # Enviar a transação para a API Flask para fazer previsões
    try:
        response = requests.post('http://localhost:5000/predict', json=[transaction])
        response.raise_for_status()  # Levanta exceções para códigos de status HTTP de erro
        prediction = response.json()  # Isso pode causar a exceção que você viu
    except requests.exceptions.RequestException as e:
        print(f"Erro na requisição: {e}")
        continue
    except json.JSONDecodeError as e:
        print(f"Erro ao decodificar JSON: {e}")
        continue

    # Ajustando o threshold
    ajustadas = ajustar_threshold(prediction['probabilities'], best_threshold)
    
    # Adicione a previsão ajustada à lista
    pred_labels.append(ajustadas[0])
    
    # Formatando a saída
    output_df = pd.DataFrame([transaction])
    output_df['Prediction'] = ajustadas
    output_df['Probability'] = prediction['probabilities']
    
    # Adicionando explicações
    output_df['Prediction'] = output_df['Prediction'].apply(lambda x: 'Fraudulenta' if x == 1.0 else 'Legítima')
    output_df['Probability'] = output_df['Probability'].apply(lambda x: f"{x:.2%}")
    
    print("Dados da Transação e Previsão:")
    print(output_df.to_string(index=False))
    print("\n----------------------\n")

# Calcular e exibir métricas de desempenho após todas as previsões
print("Relatório de Classificação:")
print(classification_report(true_labels, pred_labels, zero_division=1))
print("Matriz de Confusão:")
print(confusion_matrix(true_labels, pred_labels))
