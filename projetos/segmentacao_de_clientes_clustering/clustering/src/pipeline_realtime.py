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

from pyspark.sql import SparkSession
from kafka import KafkaConsumer
import joblib
import json

# Configuração do Spark
spark = SparkSession.builder.appName("RealTimeCustomerSegmentation").getOrCreate()

# Carregar o modelo KMeans treinado a partir do arquivo .pkl
model_path = "D:/Github/data-science/projetos/segmentacao_de_clientes_clustering/clustering/models/kmeans_model.pkl"
model = joblib.load(model_path)

# Configuração do Kafka
consumer = KafkaConsumer('customer_data', bootstrap_servers=['localhost:9092'])

# Processamento de Dados em Tempo Real
for message in consumer:
    data = json.loads(message.value.decode('utf-8'))
    df = spark.createDataFrame([data])
    
    # Preprocessamento (se necessário)
    assembler = VectorAssembler(inputCols=df.columns, outputCol="features")
    data_transformed = assembler.transform(df)
    
    # Previsões do modelo KMeans
    predictions = model.transform(data_transformed)
    predictions.show()

    # Salvar os dados segmentados (exemplo com DynamoDB)
    # Aqui você pode adicionar a lógica para salvar os dados no banco de dados ou CRM
