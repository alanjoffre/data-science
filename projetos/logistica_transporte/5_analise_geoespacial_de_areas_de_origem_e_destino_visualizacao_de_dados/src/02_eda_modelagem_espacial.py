import os
import time
import logging
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
import networkx as nx
from prophet import Prophet

# Configuração do logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Definição dos caminhos
data_path = "D:/Github/data-science/projetos/logistica_transporte/5_analise_geoespacial_de_areas_de_origem_e_destino_visualizacao_de_dados/data/processed/processado.parquet"
figures_dir = "D:/Github/data-science/projetos/logistica_transporte/5_analise_geoespacial_de_areas_de_origem_e_destino_visualizacao_de_dados/reports/figures"
os.makedirs(figures_dir, exist_ok=True)

# Inicia a medição do tempo de execução
start_time = time.time()

# Carrega o dataset processado
df = pd.read_parquet(data_path)
logging.info(f"Dataset importado: {df.shape[0]} registros")
logging.info("Colunas e tipos de dados:")
logging.info(df.dtypes)

# --- Criação de GeoDataFrames ---
# GeoDataFrame para pontos de origem e destino (assumindo CRS EPSG:4326)
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

# --- 1. Mapeamento da Distribuição Espacial ---
fig, ax = plt.subplots(figsize=(10, 8))
# Plota pontos de origem (azul) e destino (vermelho)
gdf_origin.plot(ax=ax, color='blue', markersize=2, label='Origem')
gdf_destination.plot(ax=ax, color='red', markersize=2, label='Destino')
ax.set_title("Distribuição Espacial dos Pontos de Origem e Destino")
ax.legend()
fig_path = os.path.join(figures_dir, "distribuicao_espacial.png")
plt.savefig(fig_path, dpi=150)
plt.close(fig)
logging.info(f"Mapa de distribuição espacial salvo em: {fig_path}")

# --- 2. Visualização dos Clusters (DBSCAN) para Pontos de Origem ---
fig, ax = plt.subplots(figsize=(10, 8))
gdf_origin['cluster_origem'] = df['cluster_origem']  # Já presente no dataframe processado
gdf_origin.plot(column='cluster_origem', categorical=True, legend=True, markersize=2, ax=ax)
ax.set_title("Clusters de Origem (DBSCAN)")
fig_path = os.path.join(figures_dir, "clusters_origem.png")
plt.savefig(fig_path, dpi=150)
plt.close(fig)
logging.info(f"Mapa de clusters de origem salvo em: {fig_path}")

# --- 3. Heatmap via Kernel Density Estimation (KDE) para Pontos de Origem ---
# Obtem as coordenadas dos pontos de origem
x = gdf_origin.geometry.x.values
y = gdf_origin.geometry.y.values
xy = np.vstack([x, y])
kde = gaussian_kde(xy)
# Define a grade para avaliação da densidade
xmin, xmax = x.min(), x.max()
ymin, ymax = y.min(), y.max()
xx, yy = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]
positions = np.vstack([xx.ravel(), yy.ravel()])
density = np.reshape(kde(positions).T, xx.shape)

fig, ax = plt.subplots(figsize=(10, 8))
ax.imshow(np.rot90(density), cmap=plt.cm.hot,
          extent=[xmin, xmax, ymin, ymax])
gdf_origin.plot(ax=ax, marker='.', color='blue', markersize=1, alpha=0.5)
ax.set_title("Heatmap de Densidade (KDE) - Origens")
fig_path = os.path.join(figures_dir, "heatmap_origens.png")
plt.savefig(fig_path, dpi=150)
plt.close(fig)
logging.info(f"Heatmap de densidade salvo em: {fig_path}")

# --- 4. Previsão de Fluxo e Detecção de Anomalias com Prophet ---
# Agrega as viagens por data (utilizando a data de início)
fluxo_df = df.copy()
fluxo_df['ds'] = fluxo_df['data_hora_de_inicio_da_viagem'].dt.date
fluxo_agg = fluxo_df.groupby('ds').size().reset_index(name='y')
# Ajusta o modelo Prophet
model = Prophet()
model.fit(fluxo_agg)
# Cria dataframe para previsão dos próximos 30 dias
future = model.make_future_dataframe(periods=30)
forecast = model.predict(future)
fig = model.plot(forecast)
plt.title("Previsão de Fluxo de Viagens")
fig_path = os.path.join(figures_dir, "previsao_fluxo.png")
plt.savefig(fig_path, dpi=150)
plt.close(fig)
logging.info(f"Gráfico de previsão de fluxo salvo em: {fig_path}")

# --- Exemplo de Modelagem de Rede (Graph Theory) ---
# Agrega fluxos entre localidades de origem e destino
fluxos = df.groupby(['localizacao_centro_de_partida', 'localizacao_centro_de_destino']).size().reset_index(name='contagem')
G = nx.DiGraph()
for _, row in fluxos.iterrows():
    origem = row['localizacao_centro_de_partida']
    destino = row['localizacao_centro_de_destino']
    peso = row['contagem']
    G.add_edge(origem, destino, weight=peso)

# Plot do grafo (simplificado)
fig, ax = plt.subplots(figsize=(10, 8))
pos = nx.spring_layout(G, k=0.15, iterations=20)
nx.draw_networkx_nodes(G, pos, ax=ax, node_size=50)
nx.draw_networkx_edges(G, pos, ax=ax, arrowstyle='->', arrowsize=10)
ax.set_title("Grafo de Fluxos entre Localidades")
ax.axis('off')
fig_path = os.path.join(figures_dir, "grafo_fluxos.png")
plt.savefig(fig_path, dpi=150)
plt.close(fig)
logging.info(f"Grafo de fluxos salvo em: {fig_path}")

# --- Tempo de Execução ---
end_time = time.time()
elapsed_time = end_time - start_time
logging.info(f"Tempo de execução: {elapsed_time:.2f} segundos")
