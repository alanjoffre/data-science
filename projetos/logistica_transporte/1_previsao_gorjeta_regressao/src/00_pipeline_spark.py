import pyspark
from pyspark.sql import SparkSession
from pyspark.sql.functions import to_date, col, count, when, isnan, dayofweek, hour
from pyspark.ml.feature import OneHotEncoder, StringIndexer, VectorAssembler, StandardScaler, RFormula
from pyspark.ml import Pipeline
from pyspark.ml.stat import Correlation
from pyspark.ml.linalg import Vectors
from pyspark.sql.types import BooleanType

# Configurar a sessão do Spark
spark = SparkSession.builder \
    .appName("PrevisaoGorjeta") \
    .config("spark.master", "local[*]") \
    .config("spark.executor.memory", "8g") \
    .config("spark.driver.memory", "8g") \
    .config("spark.kryoserializer.buffer.max", "1g") \
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

# Imprimir os tipos de dados de cada coluna
print("Tipos de dados das colunas:")
for column, dtype in df.dtypes:
    print(f"{column}: {dtype}")

# Converter colunas de datas para o formato correto
if 'data_inicio' in df.columns and 'data_final' in df.columns:
    df = df.withColumn('data_inicio', to_date(col('data_inicio'), 'yyyy-MM-dd')) \
           .withColumn('data_final', to_date(col('data_final'), 'yyyy-MM-dd'))

# Criar novas variáveis baseadas em datas
if 'data_inicio' in df.columns and 'data_final' in df.columns:
    df = df.withColumn('dia_da_semana_inicio', dayofweek(col('data_inicio'))) \
           .withColumn('dia_da_semana_final', dayofweek(col('data_final')))

if 'hora_inicio' in df.columns and 'hora_final' in df.columns:
    df = df.withColumn('hora_do_dia_inicio', hour(col('hora_inicio'))) \
           .withColumn('hora_do_dia_final', hour(col('hora_final')))

# Exibir as primeiras linhas do DataFrame para verificação
df.show(5)

# Verificar valores ausentes por coluna, ignorando a coluna 'viagem_compartilhada_autorizada'
null_count = {column: df.filter(col(column).isNull()).count() for column in df.columns}
nan_count = {column: df.filter(isnan(col(column))).count() for column in df.columns}

# Exibir os resultados detalhados
print("Valores nulos por coluna (Null):")
for column, count in null_count.items():
    print(f"{column}: {count} valores nulos")

print("\nValores NaN por coluna:")
for column, count in nan_count.items():
    print(f"{column}: {count} valores NaN")

# Informar quais colunas possuem valores ausentes
columns_with_missing = [column for column, count in null_count.items() if count > 0 or nan_count[column] > 0]
print("Colunas com valores ausentes (Null ou NaN):", columns_with_missing)

# Remover colunas com muitos valores ausentes (limiar: 50% do número de linhas)
threshold = 0.5 * numero_linhas_inicial
cols_to_drop = [column for column, count in null_count.items() if count > threshold or nan_count[column] > threshold]
df = df.drop(*cols_to_drop)
print(f"Colunas removidas devido a muitos valores ausentes: {cols_to_drop}")

# Imputar valores ausentes nas colunas restantes
default_values = {
    'trato_do_censo_do_embarque': 0,
    'trato_do_censo_do_desembarque': 0,
    'area_comunitaria_do_embarque': 0,
    'area_comunitaria_do_desembarque': 0,
    'tarifa': 0,
    'gorjeta': 0,
    'cobrancas_adicionais': 0,
    'total_da_viagem': 0,
    'latitude_do_centroide_do_embarque': 0,
    'longitude_do_centroide_do_embarque': 0,
    'latitude_do_centroide_do_desembarque': 0,
    'longitude_do_centroide_do_desembarque': 0,
    'viagem_compartilhada_autorizada': False  # Para variáveis booleanas
}
df = df.fillna(default_values)

# Exibir as primeiras linhas após a limpeza
df.show(5)

# Codificação de variáveis categóricas
categorical_columns = ['trato_do_censo_do_embarque', 'trato_do_censo_do_desembarque',
                       'area_comunitaria_do_embarque', 'area_comunitaria_do_desembarque',
                       'dia_da_semana_inicio', 'hora_do_dia_inicio',
                       'dia_da_semana_final', 'hora_do_dia_final']

indexers = [StringIndexer(inputCol=column, outputCol=column + "_index") for column in categorical_columns]
encoders = [OneHotEncoder(inputCol=column + "_index", outputCol=column + "_vec") for column in categorical_columns]

# Remoção de outliers utilizando Z-score
numeric_columns = ['segundos_da_viagem', 'milhas_da_viagem', 'tarifa', 'gorjeta',
                   'cobrancas_adicionais', 'total_da_viagem',
                   'latitude_do_centroide_do_embarque', 'longitude_do_centroide_do_embarque',
                   'latitude_do_centroide_do_desembarque', 'longitude_do_centroide_do_desembarque']

# Calculando Z-score para filtrar outliers
for column_name in numeric_columns:
    if column_name in df.columns:
        mean_val = df.selectExpr(f"mean({column_name}) as mean").first()["mean"]
        stddev_val = df.selectExpr(f"stddev({column_name}) as stddev").first()["stddev"]
        df = df.filter((col(column_name) >= mean_val - 3 * stddev_val) & 
                       (col(column_name) <= mean_val + 3 * stddev_val))

# Normalização e escalonamento dos dados
assembler = VectorAssembler(inputCols=[column for column in numeric_columns if column in df.columns], outputCol="features")
scaler = StandardScaler(inputCol="features", outputCol="scaled_features")

# Seleção de features e criação do pipeline
formula = RFormula(formula="gorjeta ~ .", featuresCol="features_rformula", labelCol="label")
pipeline = Pipeline(stages=indexers + encoders + [assembler, scaler, formula])

# Ajustar o pipeline aos dados
pipeline_model = pipeline.fit(df)
df_transformed = pipeline_model.transform(df)

# Exibir as primeiras linhas após o pipeline
df_transformed.select("features_rformula", "label").show(5)

# Cálculo de correlação
if 'features_corr' not in df.columns:
    assembler_corr = VectorAssembler(inputCols=numeric_columns, outputCol="features_corr")
    df_corr = assembler_corr.transform(df)
    correlation_matrix = Correlation.corr(df_corr, "features_corr").head()[0].toArray()
    print("Matriz de Correlação de Pearson:")
    print(correlation_matrix)

# Salvar os dados processados
processed_dir = "D:/Github/data-science/projetos/logistica_transporte/1_previsao_gorjeta_regressao/data/processed/"
df_transformed.select("features_rformula", "label").write.parquet(processed_dir + "dataset_processado.parquet", mode="overwrite")
df_transformed.select("features_rformula", "label").write.csv(processed_dir + "dataset_processado.csv", header=True, mode="overwrite")

print("Dataset final processado salvo com sucesso.")
