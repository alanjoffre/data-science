import pyspark
from pyspark.sql import SparkSession
from pyspark.sql.functions import to_date, col, count, when, isnan, dayofweek, hour

# Configurar a sessão do Spark para rodar no modo local com memória aumentada
spark = SparkSession.builder \
    .appName("PrevisaoGorjeta") \
    .config("spark.master", "local[*]") \
    .config("spark.executor.memory", "4g") \
    .config("spark.driver.memory", "4g") \
    .config("spark.kryoserializer.buffer.max", "512m") \
    .getOrCreate()

# Caminho do dataset
dataset_path = "D:/Github/data-science/projetos/logistica_transporte/1_previsao_gorjeta_regressao/data/raw/dataset_preprocessado.parquet"

# Importar o dataset
df = spark.read.parquet(dataset_path)

# Informar o número de linhas no dataset carregado
numero_linhas_inicial = df.count()
print(f"O dataset possui {numero_linhas_inicial} linhas inicialmente.")

# Listar todas as colunas do dataset
print("Colunas do dataset:", df.columns)

# Certifique-se de que a coluna 'data_inicio' e 'data_final' sejam convertidas para o formato de data
df = df.withColumn('data_inicio', to_date(col('data_inicio'), 'yyyy-MM-dd')) \
       .withColumn('data_final', to_date(col('data_final'), 'yyyy-MM-dd'))

# Criar novas variáveis baseadas nas datas
df = df.withColumn('dia_da_semana_inicio', dayofweek(col('data_inicio'))) \
       .withColumn('hora_do_dia_inicio', hour(col('hora_inicio'))) \
       .withColumn('dia_da_semana_final', dayofweek(col('data_final'))) \
       .withColumn('hora_do_dia_final', hour(col('hora_final')))

# Exibir as primeiras linhas do DataFrame para verificar as datas e novas colunas
df.show(5)

# Verificar valores ausentes em todas as colunas
missing_count = df.select([count(when(col(c).isNull(), c)).alias(c) for c in df.columns]).collect()[0].asDict()
print("Valores ausentes por coluna:", missing_count)

# Remover colunas com muitos valores ausentes (definido como mais de 50% de valores ausentes)
threshold = 0.5 * numero_linhas_inicial  # Definimos o limite como 50% do número de linhas
cols_to_drop = [col for col, count in missing_count.items() if count > threshold]
df = df.drop(*cols_to_drop)
print(f"Colunas removidas devido a muitos valores ausentes: {cols_to_drop}")

# Imputar valores ausentes nas colunas restantes
df = df.fillna({'trato_do_censo_do_embarque': 0, 'trato_do_censo_do_desembarque': 0, 'area_comunitaria_do_embarque': 0, 'area_comunitaria_do_desembarque': 0, 'tarifa': 0, 'gorjeta': 0, 'cobrancas_adicionais': 0, 'total_da_viagem': 0, 'latitude_do_centroide_do_embarque': 0, 'longitude_do_centroide_do_embarque': 0, 'latitude_do_centroide_do_desembarque': 0, 'longitude_do_centroide_do_desembarque': 0})

# Exibir as primeiras linhas após a limpeza
df.show(5)

from pyspark.ml.feature import OneHotEncoder, StringIndexer, VectorAssembler, StandardScaler

# Codificação de variáveis categóricas
categorical_columns = ['trato_do_censo_do_embarque', 'trato_do_censo_do_desembarque', 'area_comunitaria_do_embarque', 'area_comunitaria_do_desembarque', 'dia_da_semana_inicio', 'hora_do_dia_inicio', 'dia_da_semana_final', 'hora_do_dia_final']

indexers = []
encoders = []

for column in categorical_columns:
    if column + "_index" not in df.columns:
        indexer = StringIndexer(inputCol=column, outputCol=column + "_index")
        df = indexer.fit(df).transform(df)
        indexers.append(indexer)
    if column + "_vec" not in df.columns:
        encoder = OneHotEncoder(inputCol=column + "_index", outputCol=column + "_vec")
        df = encoder.fit(df).transform(df)
        encoders.append(encoder)

# As colunas codificadas agora estão no formato _vec

# Remover outliers utilizando Z-score
numeric_columns = ['segundos_da_viagem', 'milhas_da_viagem', 'tarifa', 'gorjeta', 'cobrancas_adicionais', 'total_da_viagem', 'latitude_do_centroide_do_embarque', 'longitude_do_centroide_do_embarque', 'latitude_do_centroide_do_desembarque', 'longitude_do_centroide_do_desembarque']

for col_name in numeric_columns:
    mean_val = df.agg({col_name: "mean"}).collect()[0][0]
    stddev_val = df.agg({col_name: "stddev"}).collect()[0][0]
    df = df.filter((col(col_name) >= mean_val - 3 * stddev_val) & (col(col_name) <= mean_val + 3 * stddev_val))

# Normalização/Escala dos Dados
assembler = VectorAssembler(inputCols=numeric_columns, outputCol="features", handleInvalid="skip")
df = assembler.transform(df)

scaler = StandardScaler(inputCol="features", outputCol="scaled_features")
df = scaler.fit(df).transform(df)

# Exibir as primeiras linhas após a codificação e remoção de outliers
df.show(5)

from pyspark.ml.feature import RFormula
from pyspark.ml import Pipeline

# Seleção de features usando RFormula
formula = RFormula(formula="gorjeta ~ .", featuresCol="features_rformula", labelCol="label")
df = formula.fit(df).transform(df)

# Criar o pipeline com todas as etapas
pipeline = Pipeline(stages=indexers + encoders + [assembler, scaler, formula])

# Ajustar o pipeline aos dados
pipeline_model = pipeline.fit(df)
df_transformed = pipeline_model.transform(df)

# Exibir as primeiras linhas do DataFrame após a transformação pelo pipeline
df_transformed.select("id_viagem", "features_rformula", "label").show(5)

from pyspark.ml.stat import Correlation
from pyspark.ml.linalg import Vectors

# Calcular a correlação entre as features e a variável alvo (gorjeta)
numeric_columns = ['segundos_da_viagem', 'milhas_da_viagem', 'tarifa', 'gorjeta', 'cobrancas_adicionais', 'total_da_viagem', 'latitude_do_centroide_do_embarque', 'longitude_do_centroide_do_embarque', 'latitude_do_centroide_do_desembarque', 'longitude_do_centroide_do_desembarque']

# Assegure-se de que a coluna 'features' esteja no formato correto para calcular a correlação
assembler = VectorAssembler(inputCols=numeric_columns, outputCol="features_corr")
df_corr = assembler.transform(df)

# Calcular a matriz de correlação de Pearson
correlation_matrix = Correlation.corr(df_corr, "features_corr").head()[0]
print("Matriz de Correlação de Pearson:\n" + str(correlation_matrix))

# Informar o número de linhas no dataset final processado
numero_linhas_final = df_transformed.count()
print(f"O dataset final possui {numero_linhas_final} linhas.")
print(f"Redução no número de linhas: {numero_linhas_inicial - numero_linhas_final}")

# Exibir as primeiras linhas do DataFrame final transformado
df_transformed.select("id_viagem", "features_rformula", "label").show(5)

# Salvar o DataFrame final em arquivos Parquet e CSV
processed_dir = "D:/Github/data-science/projetos/logistica_transporte/1_previsao_gorjeta_regressao/data/processed/"
processed_parquet_path = processed_dir + "dataset_processado.parquet"
processed_csv_path = processed_dir + "dataset_processado.csv"

df_transformed.select("id_viagem", "features_rformula", "label").write.parquet(processed_parquet_path, mode='overwrite')
df_transformed.select("id_viagem", "features_rformula", "label").write.csv(processed_csv_path, header=True)

# Informar os caminhos onde os arquivos foram salvos
print(f"\nDataset final processado salvo em:\n- {processed_parquet_path}\n- {processed_csv_path}")
