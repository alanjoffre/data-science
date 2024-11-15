# Instalar as bibliotecas necessárias
# pip install dask
# pip install nltk
# pip install emoji
# pip install pyarrow
# pip install pandas-profiling
# pip install visions
# pip install numba
# pip install vaderSentiment
# pip install pyyaml

import dask.dataframe as dd
import pandas as pd
import numpy as np
import re
import logging
import yaml
from collections import Counter
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import nltk
import emoji
from dask import delayed

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

# Baixar recursos necessários do NLTK
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
log_info('Recursos NLTK baixados.')

# Carregar o dataset do arquivo Parquet usando Dask
file_path = config['data']['raw_file_path']
df = dd.read_parquet(file_path)
log_info('Dataset carregado com sucesso.')

# Renomear as colunas
df = df.rename(columns={
    df.columns[0]: 'id',
    df.columns[1]: 'timestamp',
    df.columns[2]: 'created_at',
    df.columns[3]: 'query',
    df.columns[4]: 'user',
    df.columns[5]: 'text'
})
log_info('Colunas renomeadas com sucesso.')

# Verificação inicial das colunas e dados
log_info(f"Informações iniciais do dataset:\n{df.info()}\n{df.describe()}")
log_info(f"Colunas do dataset: {df.columns}")

# Função para tratar a coluna created_at com fusos horários desconhecidos
def parse_date(date_str):
    try:
        return pd.to_datetime(date_str, errors='coerce', utc=True).tz_convert(None)
    except:
        return pd.NaT

df['created_at'] = df['created_at'].apply(parse_date, meta=('created_at', 'datetime64[ns]'))
log_info('Coluna created_at tratada para fusos horários desconhecidos.')

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

df['sentiment'] = df['text'].apply(analyze_sentiment, meta=('sentiment', 'str'))
log_info('Análise de sentimentos aplicada.')

# Função para limpar texto conforme configurações do config.yaml
def clean_text(text):
    if config['text_cleaning']['remove_urls']:
        text = re.sub(r'http\S+', '', text)  # Remove URLs
    if config['text_cleaning']['remove_special_characters']:
        text = re.sub(r'[^a-zA-Z@\s#]', '', text)  # Remove caracteres especiais
    if config['text_cleaning']['lowercase']:
        text = text.lower()  # Converte para minúsculas
    return text

# Função para remover stopwords
stop_words = set(stopwords.words('english'))
def remove_stopwords(text):
    if config['text_cleaning']['remove_stopwords']:
        return ' '.join([word for word in text.split() if word not in stop_words])
    return text

# Função para tokenizar texto
def tokenize_text(text):
    return word_tokenize(text)

# Função para lematizar texto
lemmatizer = WordNetLemmatizer()
def lemmatize_text(tokens):
    return [lemmatizer.lemmatize(token) for token in tokens]

# Função para remover emojis usando emoji.demojize
def remove_emojis(text):
    if config['text_cleaning']['demojize']:
        return emoji.demojize(text, delimiters=(" ", " "))
    return text

# Função para processar texto usando Dask Delayed
@delayed
def process_text(text):
    text = clean_text(text)
    text = remove_stopwords(text)
    text = remove_emojis(text)
    tokens = tokenize_text(text)
    lemmatized = lemmatize_text(tokens)
    return lemmatized

# Aplicar as funções de limpeza no dataset
df['text_cleaned'] = df['text'].apply(clean_text, meta=('text_cleaned', 'str'))
df['text_cleaned'] = df['text_cleaned'].apply(remove_stopwords, meta=('text_cleaned', 'str'))
df['text_cleaned'] = df['text_cleaned'].apply(remove_emojis, meta=('text_cleaned', 'str'))
df['text_processed'] = df['text_cleaned'].apply(lambda x: process_text(x).compute(), meta=('text_processed', 'object'))
log_info('Funções de limpeza aplicadas ao dataset.')

# Tratar valores ausentes e duplicatas
df = df.dropna(subset=['text_cleaned'])
df = df.drop_duplicates(subset=['text_cleaned'])
log_info('Valores ausentes e duplicatas tratados.')

# Visualizar as mudanças
print(df.head())
log_info('Primeiras linhas do dataset após limpeza exibidas.')

# Converter o Dask DataFrame para Pandas DataFrame para análise exploratória
df_pd = df.compute()
log_info('Dataset convertido para Pandas DataFrame.')

# Salvar o dataset limpo
processed_file_path = config['data']['processed_file_path']
df_pd.to_parquet(processed_file_path, index=False)
log_info(f'Dataset limpo salvo com sucesso em {processed_file_path}.')

# Carregar o dataset do arquivo Parquet usando Dask
df = dd.read_parquet(processed_file_path)
log_info('Dataset limpo recarregado.')

# Identificar colunas com NaNs
cols_with_nan = df.columns[df.isna().any()].tolist()
log_info(f"Colunas com NaNs: {cols_with_nan}")

# Mostrar a contagem de NaNs por coluna
nan_count = df[cols_with_nan].isna().sum().compute()
log_info(f"Contagem de NaNs por coluna:\n{nan_count}")

# Tratar valores NaN nas colunas numéricas
for col in cols_with_nan:
    if df[col].dtype in [np.float64, np.int64]:
        df[col] = df[col].fillna(df[col].mean().compute())  # Preencher NaNs com a média da coluna
        log_info(f"Valores NaN na coluna {col} preenchidos com a média.")
    elif df[col].dtype == 'datetime64[ns]':
        df[col] = df[col].fillna(df[col].mode().compute()[0])  # Preencher NaNs com a moda (valor mais frequente)
        log_info(f"Valores NaN na coluna {col} preenchidos com a moda.")

# Verificar se ainda há NaNs
final_nan_count = df.isna().sum().compute()
log_info(f"Contagem de NaNs após correção:\n{final_nan_count}")

# Salvar o dataset limpo
processed_file_path = config['data']['processed_file_path']
df_pd.to_parquet(processed_file_path, index=False)
log_info(f'Dataset limpo salvo com sucesso em {processed_file_path}.')
