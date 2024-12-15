from kafka import KafkaProducer
import json
import pandas as pd

# Inicializar o produtor Kafka
producer = KafkaProducer(
    bootstrap_servers='localhost:9092',
    value_serializer=lambda v: json.dumps(v).encode('utf-8')
)

# Carregar os dados reais gerados
data_real = pd.read_csv(r'D:\Github\data-science\projetos\creditcard_fraud_detection\creditcard\data\real_data\creditcard_processed.csv')

# Enviar cada linha de dados reais como uma mensagem Kafka
for index, row in data_real.iterrows():
    message = row.to_dict()
    producer.send('transactions', value=message)

producer.flush()
print("Dados reais enviados com sucesso.")
