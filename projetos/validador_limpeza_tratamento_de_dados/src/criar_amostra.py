import pandas as pd

# Caminho do dataset original
caminho = r'D:\Github\data-science\projetos\logistica_transporte\2_analise_compartilhamento_de_viagem_classificacao\data\raw\logistica_transportadora_2018_2022.csv'
df = pd.read_csv(caminho, low_memory=False)

# Dicionário com as traduções das colunas para português, em minúsculas e sem espaços
colunas_traducao = {
    'Trip ID': 'id_da_viagem',
    'Trip Start Timestamp': 'data_hora_de_inicio_da_viagem',
    'Trip End Timestamp': 'data_hora_de_termino_da_viagem',
    'Trip Seconds': 'duracao_segundos_da_viagem',
    'Trip Miles': 'distancia_milhas_da_viagem',
    'Pickup Census Tract': 'setor_censitario_de_partida',
    'Dropoff Census Tract': 'setor_censitario_de_destino',
    'Pickup Community Area': 'area_comunitaria_de_partida',
    'Dropoff Community Area': 'area_comunitaria_de_destino',
    'Fare': 'tarifa',
    'Tip': 'gorjeta',
    'Additional Charges': 'cobrancas_adicionais',
    'Trip Total': 'total_da_viagem',
    'Shared Trip Authorized': 'viagem_compartilhada_autorizada',
    'Trips Pooled': 'viagens_compartilhadas',
    'Pickup Centroid Latitude': 'latitude_centro_de_partida',
    'Pickup Centroid Longitude': 'longitude_centro_de_partida',
    'Pickup Centroid Location': 'localizacao_centro_de_partida',
    'Dropoff Centroid Latitude': 'latitude_centro_de_destino',
    'Dropoff Centroid Longitude': 'longitude_centro_de_destino',
    'Dropoff Centroid Location': 'localizacao_centro_de_destino'
}

# Renomear as colunas
df.rename(columns=colunas_traducao, inplace=True)

# Selecionar apenas as 50000 primeiras linhas
df_amostra = df.head(50000)

# Caminhos para salvar os arquivos
caminho_csv = r'D:\Github\data-science\projetos\validador_limpeza_tratamento_de_dados\data\processed\amostra.csv'
caminho_parquet = r'D:\Github\data-science\projetos\validador_limpeza_tratamento_de_dados\data\processed\amostra.parquet'

# Salvar em CSV
df_amostra.to_csv(caminho_csv, index=False)

# Salvar em Parquet
df_amostra.to_parquet(caminho_parquet, index=False)

print("Arquivo salvo em CSV:", caminho_csv)
print("Arquivo salvo em Parquet:", caminho_parquet)
