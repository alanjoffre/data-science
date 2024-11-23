import dask.dataframe as dd
import logging
import yaml
import os
import numpy as np
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

def converter_array_para_lista(tokens):
    if isinstance(tokens, (list, np.ndarray)):
        return list(tokens)
    return tokens

def join_tokens(tokens):
    # Assegurar que cada token é uma palavra, e não caracteres individuais
    if isinstance(tokens, list) and len(tokens) > 0:
        joined_tokens = ' '.join(tokens)
        return joined_tokens
    return ''

def carregar_dados():
    try:
        stemmed_lemmatized_data_path = os.path.join(config['directories']['processed_data'], config['files']['processed_dataset'])
        df = dd.read_parquet(stemmed_lemmatized_data_path)
        logger.info(f"Dataset carregado com sucesso com {len(df)} linhas.")
        return df
    except Exception as e:
        logger.error("Erro ao carregar os dados: %s", e)
        raise

def preprocessar_dados(df):
    try:
        df['tokens_lemmatized'] = df['tokens_lemmatized'].apply(converter_array_para_lista, meta=('tokens_lemmatized', object))
        df['tokens_lemmatized'] = df['tokens_lemmatized'].apply(join_tokens, meta=('tokens_lemmatized', str))
        df = df[df['tokens_lemmatized'].str.strip().astype(bool)]
        return df
    except Exception as e:
        logger.error("Erro ao preprocessar os dados: %s", e)
        raise

def aplicar_tfidf(df):
    try:
        tfidf_documents = df['tokens_lemmatized'].compute()

        logger.info("Documentos após pré-processamento:")
        for doc in tfidf_documents.head(5):
            logger.info(doc)
        
        if tfidf_documents.empty:
            logger.error("Todos os documentos estão vazios ou contêm apenas stop words.")
            return None

        tfidf = TfidfVectorizer(token_pattern=r'\b\w+\b')
        tfidf_matrix = tfidf.fit_transform(tfidf_documents)
        
        df_tfidf = pd.DataFrame(tfidf_matrix.toarray(), columns=tfidf.get_feature_names())
        return df_tfidf
    except Exception as e:
        logger.error("Erro ao aplicar TF-IDF: %s", e)
        raise

def salvar_dados(df_tfidf, path_prefix):
    try:
        vectorized_data_path = os.path.join(path_prefix, "etapa2_6_vetorizacao_texto.parquet")
        df_tfidf.to_parquet(vectorized_data_path, index=False)

        csv_data_path = os.path.join(path_prefix, "etapa2_6_vetorizacao_texto.csv")
        df_tfidf.to_csv(csv_data_path, index=False)

        logger.info("Dataset salvo com sucesso.")
    except Exception as e:
        logger.error("Erro ao salvar os dados: %s", e)
        raise

def vetorizacao_texto():
    df = carregar_dados()
    df = preprocessar_dados(df)
    df_tfidf = aplicar_tfidf(df)
    
    if df_tfidf is not None:
        salvar_dados(df_tfidf, config['directories']['processed_data'])
        atualizar_config(config, 'files', {'log_file': "etapa2_6_vetorizacao_texto.log",
                                           'processed_dataset': "etapa2_6_vetorizacao_texto.parquet",
                                           'processed_dataset_csv': "etapa2_6_vetorizacao_texto.csv",
                                           'raw_dataset': "asentimentos.parquet"})

if __name__ == "__main__":
    vetorizacao_texto()
