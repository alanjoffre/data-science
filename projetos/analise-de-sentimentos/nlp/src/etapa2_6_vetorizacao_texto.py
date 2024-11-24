import logging
import yaml
import os
import numpy as np
import json
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd

# Carregar arquivo de configuração
config_path = 'D:\\Github\\data-science\\projetos\\analise-de-sentimentos\\nlp\\config\\config.yaml'
with open(config_path, 'r') as file:
    config = yaml.safe_load(file)

# Configuração de Logs
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[logging.StreamHandler(),
                              logging.FileHandler(os.path.join(config['directories']['logs'], config['files']['log_file']))])
logger = logging.getLogger(__name__)

def atualizar_config(config, chave, valor, config_path='D:\\Github\\data-science\\projetos\\analise-de-sentimentos\\nlp\\config\\config.yaml'):
    config[chave] = valor
    with open(config_path, 'w') as file:
        yaml.safe_dump(config, file)
    logger.info("Arquivo config.yaml atualizado com sucesso.")

def carregar_dados():
    try:
        stemmed_lemmatized_data_path = os.path.join(config['directories']['processed_data'], config['files']['processed_dataset'])
        df = pd.read_parquet(stemmed_lemmatized_data_path)
        logger.info(f"Dataset carregado com sucesso com {len(df)} linhas.")
        return df
    except Exception as e:
        logger.error("Erro ao carregar os dados: %s", e)
        raise

def preprocessar_dados(df):
    try:
        df['tweet'] = df['tweet'].astype(str)
        return df
    except Exception as e:
        logger.error("Erro ao preprocessar os dados: %s", e)
        raise

def aplicar_tfidf(df):
    try:
        tweets = df['tweet']
        
        tfidf = TfidfVectorizer(token_pattern=r'\b\w+\b')
        tfidf_matrix = tfidf.fit_transform(tweets)
        
        vetorizacao_tweets = tfidf_matrix.toarray().tolist()
        vetorizacao_tweets_json = [json.dumps(vec) for vec in vetorizacao_tweets]
        return vetorizacao_tweets_json
    except Exception as e:
        logger.error("Erro ao aplicar TF-IDF: %s", e)
        raise

def salvar_dados(df, path_prefix):
    try:
        vectorized_data_path = os.path.join(path_prefix, "etapa2_6_vetorizacao_texto.parquet")
        df.to_parquet(vectorized_data_path, index=False)

        csv_data_path = os.path.join(path_prefix, "etapa2_6_vetorizacao_texto.csv")
        df.to_csv(csv_data_path, index=False)

        logger.info("Dataset salvo com sucesso.")
    except Exception as e:
        logger.error("Erro ao salvar os dados: %s", e)
        raise

def vetorizacao_texto():
    try:
        df = carregar_dados()
        df = preprocessar_dados(df)
        
        vetorizacao_tweets = aplicar_tfidf(df)
        df['vetorizacao_tweet'] = vetorizacao_tweets

        salvar_dados(df, config['directories']['processed_data'])
        atualizar_config(config, 'files', {'log_file': "etapa2_6_vetorizacao_texto.log",
                                           'processed_dataset': "etapa2_6_vetorizacao_texto.parquet",
                                           'processed_dataset_csv': "etapa2_6_vetorizacao_texto.csv",
                                           'raw_dataset': "asentimentos.parquet"})
    except Exception as e:
        logger.error("Erro durante a vetorizacao_texto: %s", e)
        raise

if __name__ == "__main__":
    vetorizacao_texto()
