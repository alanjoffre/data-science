import dask.dataframe as dd
import logging
import yaml
import os
import spacy
from nltk.tokenize import word_tokenize
from nltk.util import ngrams
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)

# Carregar arquivo de configuração
config_path = 'D:\\Github\\data-science\\projetos\\analise-de-sentimentos\\nlp\\config\\config.yaml'
with open(config_path, 'r') as file:
    config = yaml.safe_load(file)

# Configuração de Logs
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(os.path.join(config['directories']['logs'], "etapa2_3_tokenizacao_dados.log"))
    ]
)
logger = logging.getLogger(__name__)

def atualizar_config(config, chave, valor):
    """
    Atualiza uma chave no arquivo de configuração.
    """
    if isinstance(config[chave], dict):
        config[chave].update(valor)
    else:
        config[chave] = valor

    with open(config_path, 'w') as file:
        yaml.safe_dump(config, file)
    logger.info("Arquivo config.yaml atualizado com sucesso.")

# Carregar modelo de linguagem spaCy
nlp = spacy.load("en_core_web_sm")

def tokenizar_texto(texto):
    """
    Tokeniza um texto usando spaCy, removendo pontuações.
    """
    doc = nlp(texto)
    tokens = [token.text for token in doc if not token.is_punct]  # Filtra pontuação
    return tokens  # Retorna uma lista de tokens sem pontuação

def criar_bigramas(texto):
    """
    Cria bigramas a partir de um texto.
    """
    tokens = word_tokenize(texto)
    bigrams = list(ngrams(tokens, 2))
    return ["_".join(bigram) for bigram in bigrams]

def tokenizacao_dados():
    """
    Realiza a tokenização de dados e salva o resultado.
    """
    try:
        logger.info("Carregando o dataset normalizado.")
        normalized_data_path = os.path.join(config['directories']['processed_data'], config['files']['processed_dataset'])
        df = dd.read_parquet(normalized_data_path)
        logger.info("Dataset carregado com sucesso.")

        logger.info("Tokenizando os textos.")
        df['tokens'] = df['tweet'].map_partitions(
            lambda partition: partition.map(lambda x: tokenizar_texto(x)),
            meta=('tokens', object)
        )

        logger.info("Criando bigramas.")
        df['bigrams'] = df['tweet'].map_partitions(
            lambda partition: partition.map(lambda x: criar_bigramas(x)),
            meta=('bigrams', object)
        )

        logger.info("Convertendo para pandas e ajustando tipos.")
        df = df.compute()  # Converte o Dask DataFrame para Pandas

        # Ajusta os tipos de dados
        df['tokens'] = df['tokens'].apply(lambda x: ', '.join(x) if isinstance(x, list) else x)
        df['bigrams'] = df['bigrams'].apply(lambda x: ', '.join(x) if isinstance(x, list) else x)

        logger.info("Salvando dataset tokenizado em formato Parquet e CSV.")
        tokenized_data_path = os.path.join(config['directories']['processed_data'], "etapa2_3_tokenizacao_dados.parquet")
        tokenized_data_csv_path = os.path.join(config['directories']['processed_data'], "etapa2_3_tokenizacao_dados.csv")

        df.to_parquet(tokenized_data_path, index=False)
        df.to_csv(tokenized_data_csv_path, index=False)

        logger.info("Dataset tokenizado salvo com sucesso.")
        
        # Atualizar config.yaml para a próxima etapa
        atualizar_config(config, 'files', {'log_file': "etapa2_3_tokenizacao_dados.log",
                                           'processed_dataset': "etapa2_3_tokenizacao_dados.parquet",
                                           'processed_dataset_csv': "etapa2_3_tokenizacao_dados.csv",
                                           'raw_dataset': "asentimentos.parquet"})

    except Exception as e:
        logger.error(f"Erro durante a tokenização: {e}")
        raise

if __name__ == "__main__":
    tokenizacao_dados()