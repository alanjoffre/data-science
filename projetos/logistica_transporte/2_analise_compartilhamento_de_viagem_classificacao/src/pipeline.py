import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import logging
import json
from scipy.stats import shapiro, kstest
# Habilita o IterativeImputer (MICE) do scikit-learn
from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from statsmodels.stats.outliers_influence import variance_inflation_factor

# -------------------------------------------------------------------------
# Configuração dos diretórios e caminhos dos arquivos
preprocessed_data_path = r'D:\Github\data-science\projetos\logistica_transporte\2_analise_compartilhamento_de_viagem_classificacao\data\processed\preprocessado.parquet'
output_csv_path = r'D:\Github\data-science\projetos\logistica_transporte\2_analise_compartilhamento_de_viagem_classificacao\data\processed\processado.csv'
output_parquet_path = r'D:\Github\data-science\projetos\logistica_transporte\2_analise_compartilhamento_de_viagem_classificacao\data\processed\processado.parquet'
metrics_output_path = r'D:\Github\data-science\projetos\logistica_transporte\2_analise_compartilhamento_de_viagem_classificacao\metrics\metrics.json'

# Diretório para os logs
logs_dir = r'D:\Github\data-science\projetos\logistica_transporte\2_analise_compartilhamento_de_viagem_classificacao\logs'
if not os.path.exists(logs_dir):
    os.makedirs(logs_dir)
log_file = os.path.join(logs_dir, "pipeline.log")

# Diretório para salvar os gráficos
figures_dir = r'D:\Github\data-science\projetos\logistica_transporte\2_analise_compartilhamento_de_viagem_classificacao\reports\figures'
if not os.path.exists(figures_dir):
    os.makedirs(figures_dir)

# Certifica-se de que o diretório para o JSON de métricas existe
metrics_dir = os.path.dirname(metrics_output_path)
if not os.path.exists(metrics_dir):
    os.makedirs(metrics_dir)

# -------------------------------------------------------------------------
# Configuração do logging para gravar no arquivo e na saída padrão
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)

# -------------------------------------------------------------------------
# 1. Importação do dataset
df = pd.read_parquet(preprocessed_data_path)
num_linhas_iniciais = df.shape[0]
logging.info(f"Dataset carregado com {num_linhas_iniciais} registros e {df.shape[1]} colunas.")

# -------------------------------------------------------------------------
# 2. Renomeação das colunas
colunas = {
    'Trip ID': 'id_viagem',
    'Trip Start Timestamp': 'data_inicio',
    'Trip End Timestamp': 'data_final',
    'Trip Seconds': 'segundos_da_viagem',
    'Trip Miles': 'milhas_da_viagem',
    'Pickup Census Tract': 'trato_do_censo_do_embarque',
    'Dropoff Census Tract': 'trato_do_censo_do_desembarque',
    'Pickup Community Area': 'area_comunitaria_do_embarque',
    'Dropoff Community Area': 'area_comunitaria_do_desembarque',
    'Fare': 'tarifa',
    'Tip': 'gorjeta',
    'Additional Charges': 'cobrancas_adicionais',
    'Trip Total': 'total_da_viagem',
    'Shared Trip Authorized': 'viagem_compartilhada_autorizada',
    'Trips Pooled': 'viagens_agrupadas',
    'Pickup Centroid Latitude': 'latitude_do_centroide_do_embarque',
    'Pickup Centroid Longitude': 'longitude_do_centroide_do_embarque',
    'Dropoff Centroid Latitude': 'latitude_do_centroide_do_desembarque',
    'Dropoff Centroid Longitude': 'longitude_do_centroide_do_desembarque'
}
df.rename(columns={k: v for k, v in colunas.items() if k in df.columns}, inplace=True)

# -------------------------------------------------------------------------
# 2.1. Conversão das colunas temporais para datetime
for col in ['data_inicio', 'data_final']:
    if col in df.columns:
        df[col] = pd.to_datetime(df[col], errors='coerce')

# -------------------------------------------------------------------------
# 3. Análise Exploratória Inicial (Visão Geral)
print("=== Análise Exploratória Inicial ===\n")
print("Primeiras 5 linhas do dataset:")
print(df.head(), "\n")
print("Informações do DataFrame:")
df.info()
print("\nDimensões do DataFrame (linhas, colunas):", df.shape, "\n")
print("Resumo estatístico das colunas numéricas:")
print(df.describe(), "\n")
print("Resumo estatístico das colunas categóricas:")
print(df.describe(include=['object', 'category']), "\n")
print("Contagem de valores nulos por coluna:")
print(df.isnull().sum(), "\n")
num_duplicados = df.duplicated().sum()
print("Número de registros duplicados:", num_duplicados, "\n")

# -------------------------------------------------------------------------
# 4. Conversão de Tipos para Otimização
if 'viagem_compartilhada_autorizada' in df.columns and df['viagem_compartilhada_autorizada'].dtype == 'object':
    df['viagem_compartilhada_autorizada'] = df['viagem_compartilhada_autorizada'].astype('category')
if 'tarifa' in df.columns and df['tarifa'].dtype == 'float64':
    df['tarifa'] = df['tarifa'].astype('float32')
logging.info("Após conversões, resumo do DataFrame:")
df.info()

# -------------------------------------------------------------------------
# 5. Tratamento de Dados Duplicados
num_duplicates = df.duplicated().sum()
if num_duplicates > 0:
    logging.info(f"Foram encontrados {num_duplicates} registros duplicados. Removendo-os...")
    df = df.drop_duplicates().reset_index(drop=True)
    logging.info("Registros duplicados removidos.")
else:
    logging.info("Nenhum registro duplicado encontrado.")

# -------------------------------------------------------------------------
# 6. Consistência e Padronização dos Tipos de Dados
for col in df.select_dtypes(include=['object']).columns:
    df[col] = df[col].str.strip().str.lower()
    logging.info(f"Coluna '{col}' padronizada (remoção de espaços e conversão para minúsculas).")

# -------------------------------------------------------------------------
# 7. Tratamento de Valores Inconsistentes e Outliers
for col in df.select_dtypes(include=[np.number]).columns:
    if (df[col].min() >= 0) and (df[col].skew() > 2):
        new_col = f"log_{col}"
        df[new_col] = np.log1p(df[col])
        logging.info(f"Coluna '{col}' apresenta alto skewness ({df[col].skew():.2f}). Criada coluna '{new_col}' com transformação log(x+1).")

# -------------------------------------------------------------------------
# 8. Tratamento de Dados Categóricos
for col in df.select_dtypes(include=['object']).columns:
    unique_before = df[col].nunique()
    df[col] = df[col].str.strip().str.lower()
    unique_after = df[col].nunique()
    if unique_after < unique_before:
        logging.info(f"Coluna categórica '{col}' teve redução de categorias de {unique_before} para {unique_after} após padronização.")

# -------------------------------------------------------------------------
# 9. Normalização e Escalonamento de Dados Numéricos
scaler = StandardScaler()
numeric_cols = df.select_dtypes(include=[np.number]).columns
zero_variance_cols = [col for col in numeric_cols if df[col].std() == 0]
if zero_variance_cols:
    logging.warning(f"As seguintes colunas possuem variância zero e serão removidas do escalonamento: {zero_variance_cols}")
    numeric_cols_scaler = [col for col in numeric_cols if col not in zero_variance_cols]
else:
    numeric_cols_scaler = numeric_cols

df_scaled = pd.DataFrame(scaler.fit_transform(df[numeric_cols_scaler]), 
                         columns=[f"scaled_{col}" for col in numeric_cols_scaler])
for col in zero_variance_cols:
    df_scaled[f"scaled_{col}"] = 0.0

df = pd.concat([df, df_scaled], axis=1)
logging.info("Aplicado StandardScaler nas colunas numéricas. Colunas escalonadas adicionadas com prefixo 'scaled_'.")

# -------------------------------------------------------------------------
# 10. Tratamento de Dados Temporais
if "data_inicio" in df.columns:
    df["dia_da_semana"] = df["data_inicio"].dt.dayofweek
    df["mes"] = df["data_inicio"].dt.month
    df["ano"] = df["data_inicio"].dt.year
    logging.info("Extraídas as features 'dia_da_semana', 'mes' e 'ano' da coluna 'data_inicio'.")

# -------------------------------------------------------------------------
# 11. Integração e Enriquecimento de Dados
df_externo = pd.DataFrame({
    'id_viagem': df['id_viagem'].sample(n=1000, random_state=42).unique(),
    'info_externa': np.random.choice(['alto', 'medio', 'baixo'], size=1000)
})
df = pd.merge(df, df_externo, on='id_viagem', how='left')
logging.info("Integração de dados externos concluída, coluna 'info_externa' adicionada.")

# -------------------------------------------------------------------------
# 12. Análise de Correlação e Multicolinearidade

# 12.1 Matriz de Correlação e Heatmap
numeric_cols = df.select_dtypes(include=[np.number]).columns  # Recalcula as colunas numéricas
corr_matrix = df[numeric_cols].corr()
plt.figure(figsize=(12, 10))
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="viridis")
plt.title("Heatmap de Correlação")
correlation_heatmap_path_2 = os.path.join(figures_dir, "correlation_heatmap_2.png")
plt.savefig(correlation_heatmap_path_2)
plt.close()
logging.info(f"Heatmap de correlação salvo em: {correlation_heatmap_path_2}")

# 12.2 Cálculo do Variance Inflation Factor (VIF)
# Substituir infinitos e remover linhas com NaN
df_vif = df[numeric_cols].replace([np.inf, -np.inf], np.nan).dropna()
vif_data = pd.DataFrame()
vif_data["feature"] = df_vif.columns
vif_data["VIF"] = [variance_inflation_factor(df_vif.values, i) for i in range(df_vif.shape[1])]
logging.info("VIF calculado para as variáveis numéricas:")
logging.info(vif_data.to_string(index=False))

# --- Correção: Remoção iterativa de features com VIF infinito ou acima de um limiar (ex.: 10)
vif_threshold = 10
features_to_keep = list(df_vif.columns)
while True:
    vif_data = pd.DataFrame()
    vif_data["feature"] = features_to_keep
    vif_values = []
    for i in range(len(features_to_keep)):
        vif_val = variance_inflation_factor(df_vif[features_to_keep].values, i)
        vif_values.append(vif_val)
    vif_data["VIF"] = vif_values
    max_vif = vif_data["VIF"].max()
    if np.isinf(max_vif) or max_vif > vif_threshold:
        feature_to_drop = vif_data.loc[vif_data["VIF"].idxmax(), "feature"]
        logging.warning(f"Removendo a feature '{feature_to_drop}' com VIF {max_vif:.2f} (acima do limiar {vif_threshold}).")
        features_to_keep.remove(feature_to_drop)
    else:
        break

logging.info("Features mantidas após correção do VIF:")
logging.info(vif_data.to_string(index=False))

# 12.3 Redução de Dimensionalidade com PCA
# Para evitar NaNs, removemos linhas com valores ausentes nas colunas numéricas
pca_input = df[numeric_cols].replace([np.inf, -np.inf], np.nan).dropna()
pca = PCA(n_components=min(10, len(numeric_cols)))
pca_result = pca.fit_transform(pca_input)
explained_variance = pca.explained_variance_ratio_
logging.info("Variância explicada pelos componentes do PCA:")
logging.info(explained_variance)

plt.figure(figsize=(10, 6))
plt.bar(range(1, len(explained_variance) + 1), explained_variance, color='teal')
plt.xlabel("Componentes Principais")
plt.ylabel("Variância Explicada")
plt.title("Variância Explicada por PCA")
plt.tight_layout()
pca_plot_path = os.path.join(figures_dir, "pca_variance.png")
plt.savefig(pca_plot_path)
plt.close()
logging.info(f"Gráfico de variância explicada pelo PCA salvo em: {pca_plot_path}")

# -------------------------------------------------------------------------
# 13. Detecção de Outliers e Outras Inconsistências
numeric_cols = df.select_dtypes(include=[np.number]).columns

# 13.1 Correção de discrepâncias temporais: inverte data_inicio e data_final se necessário.
if "data_inicio" in df.columns and "data_final" in df.columns:
    mask = df["data_inicio"] > df["data_final"]
    if mask.sum() > 0:
        logging.warning(f"[Correção] Encontrados {mask.sum()} registros com data_inicio > data_final. Corrigindo-os...")
        df.loc[mask, ["data_inicio", "data_final"]] = df.loc[mask, ["data_final", "data_inicio"]].values
        logging.info("Correção realizada: os valores de data_inicio e data_final foram trocados para os registros problemáticos.")
    else:
        logging.info("Nenhuma discrepância temporal encontrada para correção.")

# 13.2 Identificação de Outliers utilizando IQR e Z-score
for col in numeric_cols:
    data = df[col].dropna()
    Q1 = data.quantile(0.25)
    Q3 = data.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers_iqr = data[(data < lower_bound) | (data > upper_bound)]
    z_scores = (data - data.mean()) / data.std()
    outliers_z = data[np.abs(z_scores) > 3]
    
    logging.info(f"[Detecção de Problemas] Coluna: {col}")
    logging.info(f"  Método IQR: {len(outliers_iqr)} outliers encontrados. Limite inferior: {lower_bound:.2f}, Limite superior: {upper_bound:.2f}.")
    if not outliers_iqr.empty:
        sample_iqr = outliers_iqr.unique()[:10]
        logging.info(f"  Exemplo de valores extremos (IQR): {sample_iqr}")
    logging.info(f"  Z-score: {len(outliers_z)} outliers (|z| > 3) encontrados.")
    if not outliers_z.empty:
        sample_z = outliers_z.unique()[:10]
        logging.info(f"  Exemplo de valores extremos (Z-score): {sample_z}")

# 13.3 Verificação de valores negativos
cols_nao_negativas = ['tarifa', 'gorjeta', 'cobrancas_adicionais', 'total_da_viagem', 'segundos_da_viagem']
for col in cols_nao_negativas:
    if col in df.columns:
        negativos = df[df[col] < 0][col]
        if not negativos.empty:
            logging.warning(f"[Detecção de Problemas] Coluna '{col}' possui {len(negativos)} valores negativos: {negativos.unique()}")
        else:
            logging.info(f"[Detecção de Problemas] Coluna '{col}' não possui valores negativos.")

# 13.4 Verificação de Registros Duplicados
duplicados = df[df.duplicated()]
if not duplicados.empty:
    logging.warning(f"[Detecção de Problemas] Foram encontrados {len(duplicados)} registros duplicados.")
else:
    logging.info("[Detecção de Problemas] Não foram encontrados registros duplicados.")

logging.info("[Detecção de Problemas] Detecção preliminar concluída. Verifique os logs para detalhes dos outliers, discrepâncias e demais anomalias.")

# -------------------------------------------------------------------------
# 14. Salvamento dos Dados Processados e das Métricas
df.to_csv(output_csv_path, index=False)
df.to_parquet(output_parquet_path, index=False)

def calcular_metricas_qualidade(df):
    metrics = {}
    total_registros = len(df)
    completude = {}
    for col in df.columns:
        missing = df[col].isnull().sum()
        completude[col] = (total_registros - missing) / total_registros
    metrics['completude'] = completude

    consistencia = {}
    if 'data_inicio' in df.columns and 'data_final' in df.columns:
        consistencia['datas'] = (df['data_inicio'] <= df['data_final']).mean()
    num_checks = {}
    for col in ['segundos_da_viagem', 'milhas_da_viagem', 'tarifa', 'gorjeta', 'cobrancas_adicionais', 'total_da_viagem']:
        if col in df.columns:
            num_checks[col] = (df[col] >= 0).mean()
    consistencia['valores_numericos'] = num_checks
    metrics['consistencia'] = consistencia

    acuracia = {}
    if 'tarifa' in df.columns:
        tarifa_percentile_99 = df['tarifa'].quantile(0.99)
        acuracia['tarifa'] = (df['tarifa'] <= tarifa_percentile_99).mean()
    metrics['acuracia'] = acuracia

    atualidade = {}
    if 'data_inicio' in df.columns:
        df['data_inicio'] = pd.to_datetime(df['data_inicio'], errors='coerce')
        hoje = pd.Timestamp.now()
        ultima_data = df['data_inicio'].max()
        if pd.isnull(ultima_data):
            atualidade['atualidade_dias'] = None
            atualidade['dados_atuais'] = False
        else:
            delta = hoje - ultima_data
            atualidade['atualidade_dias'] = delta.days
            atualidade['dados_atuais'] = delta.days <= 30
    metrics['atualidade'] = atualidade

    return metrics

metricas_iniciais = calcular_metricas_qualidade(df)
with open(metrics_output_path, 'w') as f:
    json.dump(metricas_iniciais, f, indent=4, default=str)

logging.info("Pipeline executado com sucesso.")
