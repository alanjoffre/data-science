import time
import logging
import numpy as np
import pandas as pd
import yaml
import geopandas as gpd
from shapely.geometry import Point
from sklearn.impute import KNNImputer
from sklearn.cluster import DBSCAN

# Configuração do logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Carrega o arquivo de configuração
with open('D:/Github/data-science/projetos/logistica_transporte/5_analise_geoespacial_de_areas_de_origem_e_destino_visualizacao_de_dados/config/config.yaml', 'r') as f:
    config = yaml.safe_load(f)

data_path = config['data']['raw']
logging.info(f"Arquivo de dados carregado do caminho: {data_path}")

# Inicia a medição do tempo de execução
start_time = time.time()

# Carrega o dataset em formato Parquet
df = pd.read_parquet(data_path)
original_count = df.shape[0]
logging.info(f"Dataset lido: {original_count} registros")

# --- Validação e Correção de Duplicados ---
df = df.drop_duplicates()
duplicates_removed = original_count - df.shape[0]
logging.info(f"Registros duplicados removidos: {duplicates_removed}")

# --- Conversões e Otimizações Iniciais ---

# Converter colunas de data para datetime
df['data_hora_de_inicio_da_viagem'] = pd.to_datetime(df['data_hora_de_inicio_da_viagem'], errors='coerce')
df['data_hora_de_termino_da_viagem'] = pd.to_datetime(df['data_hora_de_termino_da_viagem'], errors='coerce')
logging.info("Conversão de datas realizada.")

# Converter 'id_da_viagem' para category
df['id_da_viagem'] = df['id_da_viagem'].astype('category')

# Converter colunas de códigos geoespaciais para inteiros (nullable)
cols_to_int = [
    'setor_censitario_de_partida',
    'setor_censitario_de_destino',
    'area_comunitaria_de_partida',
    'area_comunitaria_de_destino'
]
for col in cols_to_int:
    df[col] = pd.to_numeric(df[col], downcast='integer', errors='coerce').astype('Int64')
logging.info("Conversão de códigos geoespaciais realizada.")

# Converter 'viagens_compartilhadas' para inteiro, tratando valores nulos
df['viagens_compartilhadas'] = (
    pd.to_numeric(df['viagens_compartilhadas'], downcast='integer', errors='coerce')
    .fillna(0)
    .astype('int64')
)
logging.info("Conversão de viagens compartilhadas realizada.")

# --- Relatório de Dados Faltantes ---
missing_counts = df.isnull().sum()
missing_perc = (missing_counts / df.shape[0]) * 100
for col in df.columns:
    if missing_counts[col] > 0:
        logging.info(f"Coluna '{col}': {missing_counts[col]} valores faltantes ({missing_perc[col]:.2f}%).")

# --- Limpeza e Normalização ---

# 1. Remoção de registros com coordenadas inválidas (mantém registros com coordenadas dentro dos limites)
df = df[
    df['latitude_centro_de_partida'].between(-90, 90) &
    df['longitude_centro_de_partida'].between(-180, 180) &
    df['latitude_centro_de_destino'].between(-90, 90) &
    df['longitude_centro_de_destino'].between(-180, 180)
]
logging.info("Registros com coordenadas inválidas removidos.")

# 2. Imputação de dados faltantes para colunas de localização (corrigindo sem excluir)
location_cols = ['latitude_centro_de_partida', 'longitude_centro_de_partida',
                 'latitude_centro_de_destino', 'longitude_centro_de_destino']
for col in location_cols:
    median_val = df[col].median()
    df[col].fillna(median_val, inplace=True)
    logging.info(f"Coluna '{col}': valores faltantes preenchidos com mediana ({median_val}).")

# 3. Tratamento de dados faltantes em colunas numéricas via KNNImputer
num_cols = ['duracao_segundos_da_viagem', 'distancia_milhas_da_viagem',
            'tarifa', 'gorjeta', 'cobrancas_adicionais', 'total_da_viagem']
imputer = KNNImputer(n_neighbors=5)
df[num_cols] = imputer.fit_transform(df[num_cols])
logging.info("Imputação de dados faltantes realizada via KNNImputer.")

# 4. Detecção e cap de outliers usando método IQR para variáveis numéricas
def cap_outliers(series):
    Q1 = series.quantile(0.25)
    Q3 = series.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return series.clip(lower=lower_bound, upper=upper_bound)

for col in num_cols:
    df[col] = cap_outliers(df[col])
logging.info("Cap de outliers aplicado nas variáveis numéricas.")

# --- Conversão para Sistema de Referência Espacial ---

# Criação de GeoDataFrames para origem e destino com CRS WGS84 (EPSG:4326)
gdf_origin = gpd.GeoDataFrame(
    df.copy(),
    geometry=gpd.points_from_xy(df.longitude_centro_de_partida, df.latitude_centro_de_partida),
    crs="EPSG:4326"
)
gdf_destination = gpd.GeoDataFrame(
    df.copy(),
    geometry=gpd.points_from_xy(df.longitude_centro_de_destino, df.latitude_centro_de_destino),
    crs="EPSG:4326"
)
logging.info("GeoDataFrames criados com CRS EPSG:4326.")

# --- Enriquecimento e Feature Engineering ---

# Converter para sistema métrico para cálculo de distâncias (EPSG:3857)
gdf_origin_metric = gdf_origin.to_crs(epsg=3857)
gdf_destination_metric = gdf_destination.to_crs(epsg=3857)
df['distancia_calculada_m'] = gdf_origin_metric.geometry.distance(gdf_destination_metric.geometry)
logging.info("Atributo 'distancia_calculada_m' calculado.")

# Calcular tempo de deslocamento em minutos
df['tempo_deslocamento_min'] = (
    (df['data_hora_de_termino_da_viagem'] - df['data_hora_de_inicio_da_viagem'])
    .dt.total_seconds() / 60
)
logging.info("Atributo 'tempo_deslocamento_min' calculado.")

# Cálculo de buffers: reprojetar para EPSG:3857, aplicar buffer e voltar para EPSG:4326
gdf_origin_metric['buffer_100m'] = gdf_origin_metric.geometry.buffer(100)
gdf_destination_metric['buffer_100m'] = gdf_destination_metric.geometry.buffer(100)
# Reprojetar os buffers de volta para EPSG:4326, se necessário
gdf_origin['buffer_100m'] = gdf_origin_metric['buffer_100m'].to_crs(epsg=4326)
gdf_destination['buffer_100m'] = gdf_destination_metric['buffer_100m'].to_crs(epsg=4326)
logging.info("Buffers de 100 metros criados para origem e destino utilizando CRS projetado.")

# Clusterização com DBSCAN para identificar agrupamentos de pontos de origem
coords_origin = np.array(list(zip(gdf_origin_metric.geometry.x, gdf_origin_metric.geometry.y)))
dbscan_origin = DBSCAN(eps=1000, min_samples=5)
df['cluster_origem'] = dbscan_origin.fit_predict(coords_origin)
logging.info("Clusterização DBSCAN realizada para pontos de origem.")

# Clusterização para destino (opcional)
coords_destination = np.array(list(zip(gdf_destination_metric.geometry.x, gdf_destination_metric.geometry.y)))
dbscan_destination = DBSCAN(eps=1000, min_samples=5)
df['cluster_destino'] = dbscan_destination.fit_predict(coords_destination)
logging.info("Clusterização DBSCAN realizada para pontos de destino.")

# --- Indexação Espacial e Particionamento ---

# Criar índice espacial para GeoDataFrame de origem (R-tree)
sindex_origin = gdf_origin.sindex
logging.info("Índice espacial (R-tree) criado para pontos de origem.")

# Adicionar coluna de particionamento geográfico simples (por arredondamento de coordenadas)
df['regiao'] = (
    df['latitude_centro_de_partida'].round(1).astype(str) + "_" +
    df['longitude_centro_de_partida'].round(1).astype(str)
)
logging.info("Particionamento geográfico (coluna 'regiao') adicionado.")

# --- Finalização ---
end_time = time.time()
elapsed_time = end_time - start_time

# Cálculo do percentual de linhas mantidas em relação ao total importado
final_count = df.shape[0]
percentual_mantido = (final_count / original_count) * 100

# Exibe resumo do dataset e tempo de execução
print(f"Número de linhas finais: {final_count} ({percentual_mantido:.2f}% do total importado)")
print("Colunas e tipos de dados:")
print(df.dtypes)
print(f"Tempo de execução: {elapsed_time:.2f} segundos")

# --- Salvamento do Dataset Processado ---
output_csv = "D:/Github/data-science/projetos/logistica_transporte/5_analise_geoespacial_de_areas_de_origem_e_destino_visualizacao_de_dados/data/processed/processado.csv"
output_parquet = "D:/Github/data-science/projetos/logistica_transporte/5_analise_geoespacial_de_areas_de_origem_e_destino_visualizacao_de_dados/data/processed/processado.parquet"

df.to_csv(output_csv, index=False)
df.to_parquet(output_parquet, index=False)
logging.info(f"Dataset processado salvo em:\n  CSV: {output_csv}\n  Parquet: {output_parquet}")
