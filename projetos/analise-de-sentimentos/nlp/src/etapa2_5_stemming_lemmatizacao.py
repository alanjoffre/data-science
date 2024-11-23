import dask.dataframe as dd
import logging
import yaml
import os
import spacy
from nltk.stem import SnowballStemmer
import pyarrow as pa
import pyarrow.parquet as pq

# Carregar arquivo de configuração
config_path = 'D:\\Github\\data-science\\projetos\\analise-de-sentimentos\\nlp\\config\\config.yaml'
with open(config_path, 'r') as file:
    config = yaml.safe_load(file)

# Configuração de Logs
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[logging.StreamHandler(),
                              logging.FileHandler(os.path.join(config['directories']['logs'], "etapa2_5_stemming_lemmatizacao.log"))])
logger = logging.getLogger(__name__)

def atualizar_config(config, chave, valor):
    config[chave] = valor
    with open(config_path, 'w') as file:
        yaml.safe_dump(config, file)
    logger.info("Arquivo config.yaml atualizado com sucesso.")

# Carregar modelo de linguagem spaCy
nlp = spacy.load("en_core_web_sm")

# Inicializar o stemmer do NLTK
stemmer = SnowballStemmer(language='english')

def aplicar_stemming(tokens):
    return [stemmer.stem(token) for token in tokens]

def aplicar_lemmatizacao(tokens):
    doc = nlp(" ".join(tokens))
    return [token.lemma_.strip() for token in doc if token.lemma_.strip()]

def remover_documentos_vazios(df):
    try:
        logger.info("Removendo documentos vazios ou contendo apenas stop words.")
        total_linhas = len(df)
        logger.info(f"Total de linhas antes da remoção: {total_linhas}")

        df = df[df['tokens_lemmatized'].apply(lambda x: bool(x) and len(x) > 0, meta=('tokens_lemmatized', bool))]
        linhas_excluidas = total_linhas - len(df)
        
        logger.info(f"Total de linhas após a remoção: {len(df)}")
        logger.info(f"Número de linhas excluídas: {linhas_excluidas}")
        
        return df
    except Exception as e:
        logger.error("Erro ao remover documentos vazios: %s", e)
        raise

def stemming_lemmatizacao():
    try:
        logger.info("Carregando o dataset com stopwords removidas.")
        stopwords_removed_data_path = os.path.join(config['directories']['processed_data'], config['files']['processed_dataset'])
        df = dd.read_parquet(stopwords_removed_data_path)
        logger.info("Dataset carregado com sucesso.")
        
        logger.info("Aplicando stemming.")
        df['tokens_stemmed'] = df['tokens_sem_stopwords'].apply(aplicar_stemming, meta=('tokens_stemmed', object))
        
        logger.info("Aplicando lematização.")
        df['tokens_lemmatized'] = df['tokens_sem_stopwords'].apply(aplicar_lemmatizacao, meta=('tokens_lemmatized', object))
        
        df = remover_documentos_vazios(df)
        
        logger.info("Salvando dataset com stemming e lematização em formato Parquet e CSV.")
        stemmed_lemmatized_data_path = os.path.join(config['directories']['processed_data'], "etapa2_5_stemming_lemmatizacao.parquet")
        stemmed_lemmatized_data_csv_path = os.path.join(config['directories']['processed_data'], "etapa2_5_stemming_lemmatizacao.csv")
        
        df = df.compute()  # Convert Dask DataFrame to Pandas DataFrame
        df.to_csv(stemmed_lemmatized_data_csv_path, index=False)
        table = pa.Table.from_pandas(df)
        pq.write_table(table, stemmed_lemmatized_data_path)
        
        logger.info("Dataset salvo com sucesso.")
        
        # Atualizar config.yaml para a próxima etapa
        atualizar_config(config, 'files', {'log_file': "etapa2_5_stemming_lemmatizacao.log",
                                           'processed_dataset': "etapa2_5_stemming_lemmatizacao.parquet",
                                           'processed_dataset_csv': "etapa2_5_stemming_lemmatizacao.csv",
                                           'raw_dataset': "asentimentos.parquet"})
    except Exception as e:
        logger.error("Erro ao aplicar stemming e lematização: %s", e)
        raise

if __name__ == "__main__":
    stemming_lemmatizacao()
