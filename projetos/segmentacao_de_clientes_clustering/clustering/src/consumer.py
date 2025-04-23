import os

# Configurar variáveis de ambiente
os.environ['SPARK_HOME'] = "C:\\spark-3.5.3"
os.environ['PYSPARK_PYTHON'] = "C:\\Users\\alanj\\AppData\\Local\\Programs\\Python\\Python38\\python.exe"

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.ml.feature import VectorAssembler
from kafka import KafkaConsumer
import joblib
import json
from datetime import datetime

# Configuração do Spark
spark = SparkSession.builder.appName("RealTimeCustomerSegmentation").getOrCreate()

# Carregar o modelo KMeans treinado a partir do arquivo .pkl
model_path = "D:/Github/data-science/projetos/segmentacao_de_clientes_clustering/clustering/models/kmeans_model.pkl"
model = joblib.load(model_path)

# Configuração do Kafka
consumer = KafkaConsumer('customer_data', bootstrap_servers=['localhost:9092'])

# Função para corrigir os nomes das colunas e adicionar colunas faltantes
def correct_column_names(df):
    columns = {
        'Renda': 'Income',
        'Gasto em Vinhos': 'MntWines',
        'Gasto em Frutas': 'MntFruits',
        'Gasto em Produtos de Carne': 'MntMeatProducts',
        'Gasto em Produtos de Peixe': 'MntFishProducts',
        'Gasto em Produtos Doces': 'MntSweetProducts',
        'Gasto em Produtos de Ouro': 'MntGoldProds',
        'Compras em Promoções': 'NumDealsPurchases',
        'Compras pela Internet': 'NumWebPurchases',
        'Compras por Catálogo': 'NumCatalogPurchases',
        'Compras na Loja': 'NumStorePurchases',
        'Visitas ao Site por Mês': 'NumWebVisitsMonth'
    }

    for old_name, new_name in columns.items():
        if old_name in df.columns:
            df = df.withColumnRenamed(old_name, new_name)
    
    # Adicionar colunas que não estão presentes no JSON recebido
    for new_name in columns.values():
        if new_name not in df.columns:
            df = df.withColumn(new_name, F.lit(0))  # Adiciona a coluna com valor padrão 0
    
    return df

# Função para obter o contexto do cluster
def get_cluster_context(cluster):
    context = {
        0: "Clientes com renda média e altos gastos em vinhos e carne. Provavelmente estão interessados em produtos premium de vinhos e carnes.",
        1: "Clientes com renda baixa e gastos baixos em todas as categorias. Eles podem ser mais sensíveis a preços e interessados em promoções e ofertas.",
        2: "Clientes com alta renda e altos gastos em todas as categorias, especialmente vinhos e carne. Provavelmente valorizam produtos de alta qualidade e experiências premium.",
        3: "Clientes com renda alta e altos gastos em vinhos, carne e frutas. Eles podem estar interessados em produtos saudáveis e de alta qualidade.",
        4: "Clientes com renda média-baixa e gastos moderados em vinhos e carne. Eles podem apreciar ofertas e produtos com bom custo-benefício."
    }
    return context.get(cluster, "Contexto desconhecido para o cluster")

# Processamento de Dados em Tempo Real
for message in consumer:
    data = json.loads(message.value.decode('utf-8'))
    df = spark.createDataFrame([data])
    
    # Corrigir os nomes das colunas e adicionar colunas faltantes
    df = correct_column_names(df)

    # Preprocessamento (montagem do vetor de características)
    assembler = VectorAssembler(inputCols=[
        'Income', 'MntWines', 'MntFruits', 'MntMeatProducts',
        'MntFishProducts', 'MntSweetProducts', 'MntGoldProds',
        'NumDealsPurchases', 'NumWebPurchases', 'NumCatalogPurchases',
        'NumStorePurchases', 'NumWebVisitsMonth'
    ], outputCol="features")
    data_transformed = assembler.transform(df)

    # Previsões do modelo KMeans
    features = data_transformed.select("features").collect()
    predictions = model.predict([row.features for row in features])
    
    # Exibir informações detalhadas com análise e contexto do cluster
    for row, pred in zip(df.collect(), predictions):
        input_data = row.asDict()
        print("===== Previsão do Cliente =====")
        print(f"Data e Hora: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Dados de Entrada: {json.dumps(input_data, indent=4)}")
        print(f"Cluster Previsto: {pred}")
        print(f"Contexto do Cluster: {get_cluster_context(pred)}")
        print("Análise:")
        print(f"- Renda: {input_data['Income']}")
        print(f"- Gastos Totais em Produtos: {sum(input_data.get(key, 0) for key in ['MntWines', 'MntFruits', 'MntMeatProducts', 'MntFishProducts', 'MntSweetProducts', 'MntGoldProds'])}")
        print(f"- Compras Totais: {sum(input_data.get(key, 0) for key in ['NumDealsPurchases', 'NumWebPurchases', 'NumCatalogPurchases', 'NumStorePurchases'])}")
        print("-----")

    # Salvar os dados segmentados (exemplo com DynamoDB)
    # Aqui você pode adicionar a lógica para salvar os dados no banco de dados ou CRM
