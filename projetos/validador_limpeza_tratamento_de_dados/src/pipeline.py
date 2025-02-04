import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from scipy import stats
from sklearn.ensemble import IsolationForest

# ---------------------- Configurações Iniciais ---------------------- #
# Caminho do arquivo Parquet original
caminho_parquet = r'D:\Github\data-science\projetos\validador_limpeza_tratamento_de_dados\data\processed\amostra.parquet'

# Carregar o dataset
df = pd.read_parquet(caminho_parquet)
total_linhas = df.shape[0]
print("Total de linhas importadas:", total_linhas)

# ---------------------- Imputação dos Dados Faltantes ---------------------- #
# 1. Imputação para colunas numéricas utilizando KNNImputer
colunas_numericas = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
if colunas_numericas:
    imputer_knn = KNNImputer(n_neighbors=5, weights="uniform")
    df[colunas_numericas] = imputer_knn.fit_transform(df[colunas_numericas])
    print("Imputação numérica realizada com KNNImputer para as colunas:", colunas_numericas)
else:
    print("Nenhuma coluna numérica encontrada para imputação avançada.")

# 2. Imputação para colunas categóricas utilizando a moda
colunas_categoricas = df.select_dtypes(include=['object']).columns.tolist()
for coluna in colunas_categoricas:
    if df[coluna].isna().sum() > 0:
        valor_moda = df[coluna].mode()[0]
        df[coluna].fillna(valor_moda, inplace=True)
        print(f"Imputado {df[coluna].isna().sum()} valores faltantes na coluna '{coluna}' com a moda: {valor_moda}")

# ---------------------- Análise dos Dados Faltantes ---------------------- #
print("\nAnálise dos dados faltantes após a imputação:")
missing_counts = df.isna().sum()
missing_percent = (missing_counts / total_linhas) * 100
for coluna, pct in missing_percent.items():
    print(f"{coluna}: {pct:.2f}%")

# ---------------------- Identificação de Outliers (Antes do Tratamento) ---------------------- #
threshold = 3  # Limiar do Z-score
print("\nIdentificação de outliers por coluna (antes do tratamento) usando Z-score:")
for coluna in colunas_numericas:
    z = np.abs(stats.zscore(df[coluna]))
    outlier_count = np.sum(z > threshold)
    percent = (outlier_count / total_linhas) * 100
    print(f"{coluna}: {outlier_count} outliers, {percent:.2f}% do total de linhas importadas")

# Armazenar o DataFrame antes da remoção de outliers para análise de sensibilidade
df_before_outlier = df.copy()

# ---------------------- Remoção de Outliers ---------------------- #
iso = IsolationForest(contamination=0.01, random_state=42)
pred = iso.fit_predict(df[colunas_numericas])
df = df[pred == 1]
print("\nRemoção de outliers realizada com Isolation Forest:")
linhas_removidas = total_linhas - df.shape[0]
print(f"Linhas removidas: {linhas_removidas}")
print(f"Linhas restantes: {df.shape[0]}")

# ---------------------- Análise de Sensibilidade ---------------------- #
colunas_analise = ['tarifa', 'total_da_viagem', 'duracao_segundos_da_viagem']
print("\nAnálise de Sensibilidade:")
print("\nEstatísticas (Antes da remoção de outliers):")
stats_before = df_before_outlier[colunas_analise].describe()
print(stats_before)
print("\nEstatísticas (Após a remoção de outliers):")
stats_after = df[colunas_analise].describe()
print(stats_after)
print("\nDiferença nas médias (Antes - Após):")
print(stats_before.loc['mean'] - stats_after.loc['mean'])
print("\nDiferença nos desvios padrão (Antes - Após):")
print(stats_before.loc['std'] - stats_after.loc['std'])

print("\nValidação dos outliers após o tratamento (usando Z-score):")
for coluna in colunas_numericas:
    z = np.abs(stats.zscore(df[coluna]))
    outlier_count = np.sum(z > threshold)
    percent = (outlier_count / total_linhas) * 100
    print(f"{coluna}: {outlier_count} outliers, {percent:.2f}% do total de linhas importadas")

# ---------------------- Feature Engineering ---------------------- #
print("\nFeature Engineering:")

# Exemplo 1: Tarifa por milha (tarifa / distancia_milhas_da_viagem)
df['tarifa_por_milha'] = df['tarifa'] / df['distancia_milhas_da_viagem'].replace(0, np.nan)
df['tarifa_por_milha'].fillna(0, inplace=True)

# Exemplo 2: Porcentagem de gorjeta (gorjeta / tarifa)
df['porcentagem_gorjeta'] = df['gorjeta'] / df['tarifa'].replace(0, np.nan)
df['porcentagem_gorjeta'].fillna(0, inplace=True)

# Exemplo 3: Velocidade média (distancia_milhas_da_viagem / (duracao_segundos_da_viagem/3600))
df['velocidade_media'] = df['distancia_milhas_da_viagem'] / (df['duracao_segundos_da_viagem'] / 3600).replace(0, np.nan)
df['velocidade_media'].fillna(0, inplace=True)

# Exemplo 4: Razão de cobranças adicionais (cobrancas_adicionais / tarifa)
df['razao_cobrancas'] = df['cobrancas_adicionais'] / df['tarifa'].replace(0, np.nan)
df['razao_cobrancas'].fillna(0, inplace=True)

print("Novas features criadas: tarifa_por_milha, porcentagem_gorjeta, velocidade_media, razao_cobrancas")

# Gerando features polinomiais para um conjunto de variáveis (grau 2)
features_poly = ['tarifa', 'duracao_segundos_da_viagem', 'distancia_milhas_da_viagem']
poly = PolynomialFeatures(degree=2, interaction_only=False, include_bias=False)
poly_features = poly.fit_transform(df[features_poly])
# Adiciona um prefixo 'poly_' para evitar duplicação dos nomes das colunas
poly_feature_names = ["poly_" + name for name in poly.get_feature_names_out(features_poly)]
df_poly = pd.DataFrame(poly_features, columns=poly_feature_names, index=df.index)
df = pd.concat([df, df_poly], axis=1)
print("Polynomial features geradas para:", features_poly)

# ---------------------- Normalização/Padronização ---------------------- #
scaler = StandardScaler()
# Atualiza a lista de colunas numéricas (todas as variáveis numéricas, incluindo as novas features)
colunas_numericas = df.select_dtypes(include=[np.number]).columns.tolist()
df[colunas_numericas] = scaler.fit_transform(df[colunas_numericas])
print("\nPadronização aplicada às colunas numéricas:", colunas_numericas)
print("\nEstatísticas após padronização:")
print(df[colunas_numericas].describe())

# ---------------------- Salvamento do Dataset Processado ---------------------- #
caminho_csv = r'D:\Github\data-science\projetos\validador_limpeza_tratamento_de_dados\data\processed\processado.csv'
caminho_parquet_saida = r'D:\Github\data-science\projetos\validador_limpeza_tratamento_de_dados\data\processed\processado.parquet'
df.to_csv(caminho_csv, index=False)
df.to_parquet(caminho_parquet_saida, index=False)
print("\nDataset processado salvo em:")
print("CSV:", caminho_csv)
print("Parquet:", caminho_parquet_saida)
