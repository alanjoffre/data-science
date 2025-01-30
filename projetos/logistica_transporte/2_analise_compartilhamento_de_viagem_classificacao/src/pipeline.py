import polars as pl
import os
import time
import joblib
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import numpy as np

# Caminho do arquivo de entrada e saída
dataset_path = "D:/Github/data-science/projetos/logistica_transporte/2_analise_compartilhamento_de_viagem_classificacao/data/processed/preprocessado.parquet"
output_parquet = "D:/Github/data-science/projetos/logistica_transporte/2_analise_compartilhamento_de_viagem_classificacao/data/processed/processado.parquet"
output_csv = "D:/Github/data-science/projetos/logistica_transporte/2_analise_compartilhamento_de_viagem_classificacao/data/processed/processado.csv"
figures_dir = "D:/Github/data-science/projetos/logistica_transporte/2_analise_compartilhamento_de_viagem_classificacao/reports/figures/"
preprocessor_path = "D:/Github/data-science/projetos/logistica_transporte/2_analise_compartilhamento_de_viagem_classificacao/preprocessors/preprocessor.joblib"

# Cotação do dólar para real
cotacao_dolar_para_real = 6.0934

# Iniciar contagem de tempo
start_time = time.time()

# Criar diretório para figuras, se não existir
os.makedirs(figures_dir, exist_ok=True)

# Carregar os dados
df = pl.read_parquet(dataset_path)

# Número de linhas iniciais
linhas_iniciais = df.height
print(f"Número de linhas iniciais: {linhas_iniciais}")

# Renomeação das colunas para português
colunas_renomeadas = {
    "Trip ID": "id_viagem",
    "Trip Start Timestamp": "data_inicio",
    "Trip End Timestamp": "data_fim",
    "Trip Seconds": "segundos_da_viagem",
    "Trip Miles": "milhas_da_viagem",
    "Pickup Census Tract": "trato_do_censo_do_embarque",
    "Dropoff Census Tract": "trato_do_censo_do_desembarque",
    "Pickup Community Area": "area_comunitaria_do_embarque",
    "Dropoff Community Area": "area_comunitaria_do_desembarque",
    "Fare": "tarifa",
    "Tip": "gorjeta",
    "Additional Charges": "cobrancas_adicionais",
    "Trip Total": "total_da_viagem",
    "Shared Trip Authorized": "viagem_compartilhada_autorizada",
    "Trips Pooled": "viagens_agrupadas",
    "Pickup Centroid Latitude": "latitude_do_centroide_do_embarque",
    "Pickup Centroid Longitude": "longitude_do_centroide_do_embarque",
    "Pickup Centroid Location": "local_do_centroide_do_embarque",
    "Dropoff Centroid Latitude": "latitude_do_centroide_do_desembarque",
    "Dropoff Centroid Longitude": "longitude_do_centroide_do_desembarque",
    "Dropoff Centroid Location": "local_do_centroide_do_desembarque"
}

df = df.rename(colunas_renomeadas)

# Tratamento de valores ausentes
modo_embarque = df["trato_do_censo_do_embarque"].mode()
modo_desembarque = df["trato_do_censo_do_desembarque"].mode()

preprocessor = {
    "modo_embarque": modo_embarque,
    "modo_desembarque": modo_desembarque,
    "media_latitude_embarque": df["latitude_do_centroide_do_embarque"].mean(),
    "media_longitude_embarque": df["longitude_do_centroide_do_embarque"].mean(),
    "media_latitude_desembarque": df["latitude_do_centroide_do_desembarque"].mean(),
    "media_longitude_desembarque": df["longitude_do_centroide_do_desembarque"].mean()
}

joblib.dump(preprocessor, preprocessor_path)

df = df.with_columns([
    pl.col("trato_do_censo_do_embarque").fill_null(modo_embarque),
    pl.col("trato_do_censo_do_desembarque").fill_null(modo_desembarque),
    pl.col("latitude_do_centroide_do_embarque").fill_null(preprocessor["media_latitude_embarque"]),
    pl.col("longitude_do_centroide_do_embarque").fill_null(preprocessor["media_longitude_embarque"]),
    pl.col("latitude_do_centroide_do_desembarque").fill_null(preprocessor["media_latitude_desembarque"]),
    pl.col("longitude_do_centroide_do_desembarque").fill_null(preprocessor["media_longitude_desembarque"])
])

# Remoção de outliers
limite_distancia = 48.28
df = df.filter(df["milhas_da_viagem"] <= limite_distancia)

# Remoção de duplicatas
duplicatas_removidas = df.duplicated().sum()
df = df.unique()
print(f"Número de registros duplicados removidos: {duplicatas_removidas}")

# Winsorization para tarifas extremas
percentil_99 = df["tarifa"].quantile(0.99)
df = df.with_columns([
    pl.when(df["tarifa"] > percentil_99).then(percentil_99).otherwise(df["tarifa"]).alias("tarifa")
])

# Engenharia de Features
print("✅ Engenharia de Features iniciada")
df = df.with_columns([
    (pl.col("milhas_da_viagem") * 1.60934).alias("quilometros_da_viagem"),
    (pl.col("tarifa") * cotacao_dolar_para_real).alias("tarifa_reais")
])

# Salvar dataset
print("✅ Salvando dataset processado")
df.write_parquet(output_parquet)
df.write_csv(output_csv)

# Finalização
tempo_execucao = time.time() - start_time
horas, resto = divmod(tempo_execucao, 3600)
minutos, segundos = divmod(resto, 60)
print(f"✅ Processamento concluído em: {int(horas):02} horas {int(minutos):02} min e {int(segundos):02} seg")
print(f"✅ Dados salvos em:")
print(f"- {output_parquet}")
print(f"- {output_csv}")
print(f"📌 Gráfico salvo em: {figures_dir}/distribuicao_embarques.png")
