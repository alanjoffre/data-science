from pyspark.sql import SparkSession
from pyspark.sql.functions import regexp_extract, col, to_date, to_timestamp, concat_ws
import yaml

# Criação de uma SparkSession
spark = SparkSession.builder \
    .appName("LogisticaTransporte") \
    .getOrCreate()

# Carregando as configurações do arquivo YAML
with open('D:/Github/data-science/projetos/logistica_transporte/1_previsao_gorjeta_regressao/config/config.yaml', 'r') as file:
    config = yaml.safe_load(file)

# Caminho para o dataset bruto
dataset_path = config['data']['raw']

# Leitura do dataset
df = spark.read.csv(dataset_path, header=True, inferSchema=True)

# Exibindo as primeiras linhas para verificação
df.show(5)

# Renomeando as colunas conforme a convenção
df = df.withColumnRenamed('Trip ID', 'id_viagem') \
    .withColumnRenamed('Trip Start Timestamp', 'horario_inicio_viagem') \
    .withColumnRenamed('Trip End Timestamp', 'horario_fim_viagem') \
    .withColumnRenamed('Trip Seconds', 'segundo_da_viagem') \
    .withColumnRenamed('Trip Miles', 'milhas_viagem') \
    .withColumnRenamed('Pickup Census Tract', 'area_censitaria_origem') \
    .withColumnRenamed('Dropoff Census Tract', 'area_censitaria_destino') \
    .withColumnRenamed('Pickup Community Area', 'area_comunitaria_origem') \
    .withColumnRenamed('Dropoff Community Area', 'area_comunitaria_destino') \
    .withColumnRenamed('Fare', 'tarifa') \
    .withColumnRenamed('Tip', 'gorjeta') \
    .withColumnRenamed('Additional Charges', 'cobrancas_adicionais') \
    .withColumnRenamed('Trip Total', 'total_viagem') \
    .withColumnRenamed('Shared Trip Authorized', 'viagem_compartilhada_autorizada') \
    .withColumnRenamed('Trips Pooled', 'viagens_agrupadas') \
    .withColumnRenamed('Pickup Centroid Latitude', 'latitude_centroide_origem') \
    .withColumnRenamed('Pickup Centroid Longitude', 'longitude_centroide_origem') \
    .withColumnRenamed('Pickup Centroid Location', 'localizacao_centroide_origem') \
    .withColumnRenamed('Dropoff Centroid Latitude', 'latitude_centroide_destino') \
    .withColumnRenamed('Dropoff Centroid Longitude', 'longitude_centroide_destino') \
    .withColumnRenamed('Dropoff Centroid Location', 'localizacao_centroide_destino')

# Separando a data e a hora das colunas de início
df = df.withColumn('data_inicio_viagem', regexp_extract(col('horario_inicio_viagem'), r'(\d{2}/\d{2}/\d{4})', 1)) \
    .withColumn('hora_inicio_viagem', regexp_extract(col('horario_inicio_viagem'), r'(\d{2}:\d{2}:\d{2} \w{2})', 1))

# Separando a data e a hora das colunas de fim
df = df.withColumn('data_fim_viagem', regexp_extract(col('horario_fim_viagem'), r'(\d{2}/\d{2}/\d{4})', 1)) \
    .withColumn('hora_fim_viagem', regexp_extract(col('horario_fim_viagem'), r'(\d{2}:\d{2}:\d{2} \w{2})', 1))

# Convertendo a coluna 'data_inicio_viagem' e 'data_fim_viagem' para tipo Date
df = df.withColumn('data_inicio_viagem', to_date(col('data_inicio_viagem'), 'MM/dd/yyyy')) \
    .withColumn('data_fim_viagem', to_date(col('data_fim_viagem'), 'MM/dd/yyyy'))

# Corrigindo a conversão para hora, mantendo a data original e ajustando a hora
df = df.withColumn('hora_inicio_viagem', to_timestamp(concat_ws(' ', col('data_inicio_viagem'), col('hora_inicio_viagem')), 'yyyy-MM-dd hh:mm:ss a')) \
    .withColumn('hora_fim_viagem', to_timestamp(concat_ws(' ', col('data_fim_viagem'), col('hora_fim_viagem')), 'yyyy-MM-dd hh:mm:ss a'))

# Verificando as primeiras linhas após as transformações
df.select('data_inicio_viagem', 'hora_inicio_viagem', 'data_fim_viagem', 'hora_fim_viagem').show(10)

# Aplicando o filtro de data
df_filtered = df.filter((col('data_inicio_viagem') >= '2018-01-01') & (col('data_inicio_viagem') <= '2018-06-30'))

# Verificando as primeiras linhas após aplicar o filtro
df_filtered.show(10)

# Exibindo o schema do DataFrame
df_filtered.printSchema()

# Informações do dataset
df_filtered.describe().show()

# Diretório para salvar os datasets processados
processed_csv_path = 'D:/Github/data-science/projetos/logistica_transporte/1_previsao_gorjeta_regressao/data/processed/processed.csv'
processed_parquet_path = 'D:/Github/data-science/projetos/logistica_transporte/1_previsao_gorjeta_regressao/data/processed/processed.parquet'

# Salvando como CSV
df_filtered.write.csv(processed_csv_path, header=True, mode='overwrite')

# Salvando como Parquet
df_filtered.write.parquet(processed_parquet_path, mode='overwrite')
