# Primeira parte

# Instalar as bibliotecas necessárias
# pip install dask
# pip install nltk
# pip install emoji
# pip install pyarrow
# pip install matplotlib
# pip install seaborn
# pip install pyyaml

import dask.dataframe as dd
import pandas as pd
import numpy as np
import logging
import yaml
import re

# Carregar as configurações do arquivo config.yaml
with open('D:\\Github\\data-science\\projetos\\analise-de-sentimentos\\nlp\\config\\config.yaml', 'r') as file:
    config = yaml.safe_load(file)

# Configurar logging
logging.basicConfig(filename=config['logging']['filename'], level=config['logging']['level'],
                    format=config['logging']['format'])

# Função para configurar o logger
def log_info(message):
    print(message)
    logging.info(message)

log_info('Início do processamento de dados.')

# Carregar o dataset do arquivo Parquet usando Dask
file_path = config['data']['processed_file_path']
df = dd.read_parquet(file_path)
log_info('Dataset carregado com sucesso.')

# Verificação inicial das colunas e dados
print(df.head())
print(df.columns)

# Converter a coluna 'created_at' para datetime
df['created_at'] = dd.to_datetime(df['created_at'])

# Contar valores NaN na coluna 'created_at' antes do tratamento
num_nan_before = df['created_at'].isna().sum().compute()
log_info(f"Número de valores NaN na coluna 'created_at' antes do tratamento: {num_nan_before}")

# Converter 'created_at' para timestamp numérico (segundos desde a época)
df['created_at_numeric'] = df['created_at'].map_partitions(lambda x: x.astype('int64') // 10**9)

# Calcular a média da coluna 'created_at_numeric'
mean_date_numeric = df['created_at_numeric'].mean().compute()

# Substituir NaN na coluna 'created_at_numeric' pela média calculada
df['created_at_numeric'] = df['created_at_numeric'].fillna(mean_date_numeric)

# Converter 'created_at_numeric' de volta para datetime
df['created_at'] = dd.to_datetime(df['created_at_numeric'] * 10**9)

# Contar valores NaN na coluna 'created_at' após o tratamento
num_nan_after = df['created_at'].isna().sum().compute()
log_info(f"Número de valores NaN na coluna 'created_at' após o tratamento: {num_nan_after}")

# Verificar as primeiras linhas para garantir que os NaN foram tratados
print(df.head())

# Calcular a média e desvio padrão dos timestamps numéricos
mean_date = pd.to_datetime(mean_date_numeric * 10**9)
std_date_numeric = df['created_at_numeric'].std().compute()

log_info(f"Média da coluna 'created_at': {mean_date}")
log_info(f"Desvio padrão dos timestamps da coluna 'created_at': {std_date_numeric}")

# Remover a coluna 'created_at_numeric' após cálculo do desvio padrão
df = df.drop(columns=['created_at_numeric'])
log_info('Tratamento de NaN na coluna created_at concluído.')

# Validação final para assegurar que não existem NaN na coluna 'created_at'
assert df['created_at'].isna().sum().compute() == 0, "Ainda existem valores NaN na coluna 'created_at'"

log_info('Etapa 1 concluída com sucesso.')

import seaborn as sns
import matplotlib.pyplot as plt

log_info('Início da análise exploratória dos dados.')

# Análise exploratória manual
num_rows = df.shape[0].compute()
num_cols = df.shape[1]
head_data = df.head()
description = df.describe().compute()

log_info(f"Número de linhas: {num_rows}")
log_info(f"Número de colunas: {num_cols}")
log_info(f"Primeiras 5 linhas:\n{head_data}")
log_info(f"Descrição estatística:\n{description}")

log_info('É esperado sair NaN para os campos: Mean e Std (Coluna - Data)')

# Converter a coluna 'created_at' para string para evitar erros no salvamento
description['created_at'] = description['created_at'].astype(str)

# Salvar as saídas em arquivos Parquet e CSV
reports_dir = config['data']['reports_dir']
head_data.to_parquet(reports_dir + 'asentimentos_head.parquet', index=False)
description.to_parquet(reports_dir + 'asentimentos_description.parquet', index=False)
head_data.to_csv(reports_dir + 'asentimentos_head.csv', index=False)
description.to_csv(reports_dir + 'asentimentos_description.csv', index=False)
log_info('Análise exploratória concluída e resultados salvos.')

# Visualização dos dados
plt.figure(figsize=(10, 6))
sns.histplot(df['created_at'].compute(), kde=True, bins=config['exploratory_analysis']['hist_bins'])
plt.title('Distribuição de Datas')
plt.xlabel('Data')
plt.ylabel('Frequência')
plt.savefig(config['data']['figures_dir'] + config['exploratory_analysis']['date_dist_figure'])
log_info('Visualização da distribuição de datas salva.')

plt.figure(figsize=(10, 6))
sns.countplot(y='sentiment', data=df.compute(), order=df['sentiment'].value_counts().index)
plt.title('Contagem de Sentimentos')
plt.xlabel('Contagem')
plt.ylabel('Sentimento')
plt.savefig(config['data']['figures_dir'] + config['exploratory_analysis']['sentiment_count_figure'])
log_info('Visualização da contagem de sentimentos salva.')

log_info('Etapa 2 concluída com sucesso.')

import re
from collections import Counter

log_info('Início da extração e análise de hashtags e menções.')

# Função para identificar as palavras mais frequentes
def most_common_words(text, n=20):
    words = ' '.join(text).split()
    counter = Counter(words)
    most_common = counter.most_common(n)
    return most_common

# Função para extrair hashtags
def extract_hashtags(text):
    return re.findall(r"#(\w+)", text)

# Função para extrair menções
def extract_mentions(text):
    return re.findall(r"@(\w+)", text)

# Aplicar extração de hashtags e menções
df = df.assign(hashtags=df['text'].apply(extract_hashtags, meta=('hashtags', 'object')))
df = df.assign(mentions=df['text'].apply(extract_mentions, meta=('mentions', 'object')))
log_info('Hashtags e menções extraídas com sucesso.')

# Converter para Pandas DataFrame
df_pd = df.compute()
log_info('Conversão do Dask DataFrame para Pandas DataFrame concluída.')

# Contar hashtags mais frequentes
all_hashtags = [hashtag for hashtags_list in df_pd['hashtags'] for hashtag in hashtags_list]
common_hashtags = most_common_words(all_hashtags, n=config['exploratory_analysis']['top_n_hashtags'])
common_hashtags_df = pd.DataFrame(common_hashtags, columns=['hashtag', 'count'])
common_hashtags_df.to_parquet(config['data']['reports_dir'] + 'common_hashtags.parquet', index=False)
common_hashtags_df.to_csv(config['data']['reports_dir'] + 'common_hashtags.csv', index=False)
log_info('Hashtags mais frequentes salvas com sucesso.')

# Contar menções mais frequentes
all_mentions = [mention for mentions_list in df_pd['mentions'] for mention in mentions_list]
common_mentions = most_common_words(all_mentions, n=config['exploratory_analysis']['top_n_mentions'])
common_mentions_df = pd.DataFrame(common_mentions, columns=['mention', 'count'])
common_mentions_df.to_parquet(config['data']['reports_dir'] + 'common_mentions.parquet', index=False)
common_mentions_df.to_csv(config['data']['reports_dir'] + 'common_mentions.csv', index=False)
log_info('Menções mais frequentes salvas com sucesso.')

# Visualizações de Hashtags e Menções
plt.figure(figsize=(10, 5))
sns.barplot(data=common_hashtags_df, x='count', y='hashtag', palette='viridis')
plt.title('Top 20 Hashtags')
plt.savefig(config['data']['figures_dir'] + config['exploratory_analysis']['top_hashtags_figure'])
log_info('Visualização das Top 20 Hashtags salva com sucesso.')

plt.figure(figsize=(10, 5))
sns.barplot(data=common_mentions_df, x='count', y='mention', palette='viridis')
plt.title('Top 20 Menções')
plt.savefig(config['data']['figures_dir'] + config['exploratory_analysis']['top_mentions_figure'])
log_info('Visualização das Top 20 Menções salva com sucesso.')

log_info('Etapa 3 concluída com sucesso.')

log_info('Início da análise de bigrams e trigrams.')

# Função para identificar os bigrams mais comuns
def most_common_bigrams(text, n=20):
    words = ' '.join(text).split()
    bigrams = list(zip(words, words[1:]))
    bigram_counter = Counter(bigrams)
    most_common = bigram_counter.most_common(n)
    return most_common

# Função para identificar os trigrams mais comuns
def most_common_trigrams(text, n=20):
    words = ' '.join(text).split()
    trigrams = list(zip(words, words[1:], words[2:]))
    trigram_counter = Counter(trigrams)
    most_common = trigram_counter.most_common(n)
    return most_common

# Aplicar análise de bigrams e trigrams
common_bigrams = most_common_bigrams(df_pd['text_cleaned'], n=config['exploratory_analysis']['top_n_bigrams'])
common_bigrams_df = pd.DataFrame(common_bigrams, columns=['bigram', 'count'])
common_bigrams_df['bigram'] = common_bigrams_df['bigram'].apply(lambda x: ' '.join(x))

common_trigrams = most_common_trigrams(df_pd['text_cleaned'], n=config['exploratory_analysis']['top_n_trigrams'])
common_trigrams_df = pd.DataFrame(common_trigrams, columns=['trigram', 'count'])
common_trigrams_df['trigram'] = common_trigrams_df['trigram'].apply(lambda x: ' '.join(x))

common_bigrams_df.to_parquet(config['data']['reports_dir'] + 'common_bigrams.parquet', index=False)
common_bigrams_df.to_csv(config['data']['reports_dir'] + 'common_bigrams.csv', index=False)
log_info('Bigrams mais comuns salvos com sucesso.')

common_trigrams_df.to_parquet(config['data']['reports_dir'] + 'common_trigrams.parquet', index=False)
common_trigrams_df.to_csv(config['data']['reports_dir'] + 'common_trigrams.csv', index=False)
log_info('Trigrams mais comuns salvos com sucesso.')

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

log_info('Início da análise de sentimentos.')

# Função para analisar sentimentos usando VADER
analyzer = SentimentIntensityAnalyzer()

def analyze_sentiment(text):
    score = analyzer.polarity_scores(text)
    compound = score['compound']
    if compound >= 0.05:
        return 'positive'
    elif compound <= -0.05:
        return 'negative'
    else:
        return 'neutral'

# Aplicar análise de sentimentos
df = df.assign(sentiment=df['text'].apply(analyze_sentiment, meta=('sentiment', 'str')))
log_info('Análise de sentimentos aplicada com sucesso.')

# Converter para Pandas DataFrame para salvar resultados
df_pd = df.compute()
log_info('Conversão do Dask DataFrame para Pandas DataFrame concluída.')

# Salvar resultados de análise de sentimentos em arquivos Parquet e CSV
sentiment_analysis_file = config['exploratory_analysis']['sentiment_analysis_file']
sentiment_count_csv = config['exploratory_analysis']['sentiment_count_csv']
df_pd.to_parquet(config['data']['processed'] + sentiment_analysis_file, index=False)
df_pd.to_csv(config['data']['processed'] + sentiment_count_csv, index=False)
log_info('Resultados da análise de sentimentos salvos com sucesso.')

# Visualização da Análise de Sentimentos
plt.figure(figsize=(10, 6))
sns.countplot(y='sentiment', data=df_pd, order=df_pd['sentiment'].value_counts().index, palette='viridis')
plt.title('Distribuição de Sentimentos')
plt.xlabel('Contagem')
plt.ylabel('Sentimento')
plt.savefig(config['data']['figures_dir'] + config['exploratory_analysis']['sentiment_count_figure'])
log_info('Visualização da distribuição de sentimentos salva com sucesso.')

log_info('Etapa 5 concluída com sucesso.')

log_info('Início da limpeza final, verificação e salvamento dos dados.')

# Garantir que todos os dados estejam limpos e salvos corretamente

# Recarregar o DataFrame limpo para verificação final
df_final = pd.read_parquet(config['data']['processed'] + sentiment_analysis_file)
log_info('DataFrame recarregado com sucesso para verificação final.')

# Verificar as primeiras linhas e as colunas do DataFrame
print(df_final.head())
print(df_final.columns)

# Verificar se não há valores NaN no DataFrame final
nan_count = df_final.isna().sum()
log_info(f"Contagem de valores NaN no DataFrame final:\n{nan_count}")

# Salvar o DataFrame final em novos arquivos Parquet e CSV para garantir que todas as mudanças foram aplicadas
df_final.to_parquet(config['data']['final_file_path'], index=False)
df_final.to_csv(config['data']['processed'] + 'final_asentiments.csv', index=False)
log_info('DataFrame final salvo com sucesso em arquivos Parquet e CSV.')

log_info('Etapa 6 concluída com sucesso.')


