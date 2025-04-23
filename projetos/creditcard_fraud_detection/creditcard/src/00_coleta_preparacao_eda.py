import pandas as pd
import yaml
import logging
import time
from sklearn.preprocessing import StandardScaler

# Registrar a hora inicial do processamento
start_time = time.time()

# Configurar logging
with open(r'D:\Github\data-science\projetos\creditcard_fraud_detection\creditcard\config\config.yaml', 'r') as file:
    config = yaml.safe_load(file)

logging.basicConfig(filename=config['paths']['logs_path'] + 'data_preparation.log', level=logging.INFO, format='%(asctime)s - %(message)s')

logging.info("Carregando os dados do CSV.")
df = pd.read_csv(config['data']['raw_data_path'])

logging.info("Primeiras linhas do dataframe:")
logging.info(df.head())

logging.info("Analisando a estrutura dos dados:")
logging.info(df.info())

logging.info("Estatísticas descritivas:")
logging.info(df.describe())

logging.info("Verificando valores ausentes:")
logging.info(df.isnull().sum())

df = df.fillna(df.mean())
logging.info("Preenchendo valores ausentes com a média.")

df = df.drop_duplicates()
logging.info("Removendo duplicatas.")

logging.info("Normalizando os dados.")
features = df.drop(columns=['Class'])
scaler = StandardScaler()
features_normalized = scaler.fit_transform(features)
df_normalized = pd.DataFrame(features_normalized, columns=features.columns)
df_normalized['Class'] = df['Class']

logging.info("Salvando o dataset processado.")
df_normalized.to_csv(config['data']['processed_data_path'], index=False)

# Registrar a hora final do processamento
end_time = time.time()
total_time = end_time - start_time
hours, remainder = divmod(total_time, 3600)
minutes, seconds = divmod(remainder, 60)
logging.info(f"Tempo total de processamento: {int(hours)} horas, {int(minutes)} minutos e {int(seconds)} segundos.")

print("1 - Etapa de coleta e preparação de dados concluída.")

import pandas as pd
import yaml
import logging
import matplotlib.pyplot as plt
import seaborn as sns
import time

# Registrar a hora inicial do processamento
start_time = time.time()

# Configurar logging
with open(r'D:\Github\data-science\projetos\creditcard_fraud_detection\creditcard\config\config.yaml', 'r') as file:
    config = yaml.safe_load(file)

logging.basicConfig(filename=config['paths']['logs_path'] + 'eda.log', level=logging.INFO, format='%(asctime)s - %(message)s')

# Carregar os dados processados
df = pd.read_csv(config['data']['processed_data_path'])

# Contagem de classes
class_counts = df['Class'].value_counts()
logging.info("Contagem de classes:")
logging.info(class_counts)

# Gráfico de barras para distribuição das classes
plt.figure(figsize=(8, 6))
sns.countplot(x='Class', data=df)
plt.title('Distribuição das Classes')
plt.xlabel('Class')
plt.ylabel('Contagem')
plt.savefig(config['paths']['figures_path'] + 'distribuicao_classes.png')
plt.close()

# Estatísticas descritivas do valor das transações
logging.info("Estatísticas descritivas do valor das transações:")
logging.info(df['Amount'].describe())

# Histograma do valor das transações
plt.figure(figsize=(10, 6))
sns.histplot(df['Amount'], bins=50, kde=True)
plt.title('Distribuição do Valor das Transações')
plt.xlabel('Valor ($)')
plt.ylabel('Frequência')
plt.savefig(config['paths']['figures_path'] + 'distribuicao_valor_transacoes.png')
plt.close()

# Estatísticas descritivas do tempo das transações
logging.info("Estatísticas descritivas do tempo das transações:")
logging.info(df['Time'].describe())

# Gráfico de linha do tempo das transações
plt.figure(figsize=(10, 6))
sns.lineplot(x='Time', y='Amount', data=df)
plt.title('Valor das Transações ao Longo do Tempo')
plt.xlabel('Tempo (segundos)')
plt.ylabel('Valor ($)')
plt.savefig(config['paths']['figures_path'] + 'valor_transacoes_tempo.png')
plt.close()

# Matriz de correlação
corr_matrix = df.corr()
logging.info("Matriz de correlação:")
logging.info(corr_matrix)

# Heatmap da correlação
plt.figure(figsize=(20, 10))
sns.heatmap(corr_matrix, annot=False, cmap='coolwarm')
plt.title('Matriz de Correlação das Variáveis')
plt.savefig(config['paths']['figures_path'] + 'matriz_correlacao.png')
plt.close()

# Estatísticas descritivas dos valores por classe
logging.info("Estatísticas descritivas dos valores por classe:")
logging.info(df.groupby('Class')['Amount'].describe())

# Boxplot para verificar outliers
plt.figure(figsize=(10, 6))
sns.boxplot(x='Class', y='Amount', data=df)
plt.title('Boxplot do Valor das Transações por Classe')
plt.xlabel('Class')
plt.ylabel('Valor ($)')
plt.savefig(config['paths']['figures_path'] + 'boxplot_valor_transacoes_classe.png')
plt.close()

# Salvar o dataset processado
df.to_csv(config['data']['processed_data_path'], index=False)
logging.info("Dataset processado salvo em creditcard_processed.csv")

# Registrar a hora final do processamento
end_time = time.time()
total_time = end_time - start_time
hours, remainder = divmod(total_time, 3600)
minutes, seconds = divmod(remainder, 60)
logging.info(f"Tempo total de processamento: {int(hours)} horas, {int(minutes)} minutos e {int(seconds)} segundos.")

print("2 - Etapa de Análise Exploratória de Dados concluída (EDA).")

