import logging
import yaml
import dask.dataframe as dd
import pandas as pd
import spacy

# Carregar arquivo de configuração
config_path = 'D:\\Github\\data-science\\projetos\\analise-de-sentimentos\\nlp\\config\\config.yaml'
with open(config_path, 'r') as file:
    config = yaml.safe_load(file)

# Configuração de Logs
logging.basicConfig(level=config['logs']['level'],
                    format=config['logs']['format'],
                    handlers=[logging.StreamHandler(),
                              logging.FileHandler(config['directories']['logs'] + 'tokenization.log')])
logger = logging.getLogger(__name__)

# Carregar o modelo de linguagem do spaCy
nlp = spacy.load('en_core_web_sm', disable=['ner', 'parser'])

def tokenizar_texto(texto):
    doc = nlp(texto)
    return [token.text for token in doc]

def tokenizacao_texto():
    logger.info("Iniciando a etapa de tokenização de texto...")

    # Carregar o dataset processado da etapa anterior
    processed_data_path = config['directories']['processed_data'] + config['files']['processed_dataset']
    df = dd.read_parquet(processed_data_path, engine='pyarrow')

    # Tokenizar o texto dos tweets
    df['tweet_tokens'] = df['tweet'].map_partitions(lambda s: s.apply(tokenizar_texto), meta=('tweet_tokens', 'object'))

    # Converter Dask DataFrame para Pandas DataFrame
    tokenized_df = df.compute()

    # Salvar o dataset tokenizado como um único arquivo Parquet
    tokenized_data_path = config['directories']['processed_data'] + 'tokenization.parquet'
    tokenized_df.to_parquet(tokenized_data_path, engine='pyarrow', index=False)
    tokenized_df.to_csv(config['directories']['processed_data'] + 'tokenization.csv', index=False)

    # Verifique se o arquivo Parquet foi salvo corretamente
    try:
        _ = pd.read_parquet(tokenized_data_path, engine='pyarrow')
        logger.info("Arquivo Parquet salvo e lido com sucesso.")
    except Exception as e:
        logger.error(f"Erro ao salvar ou ler o arquivo Parquet: {e}")

    # Atualizar o arquivo de configuração
    config['files']['processed_dataset'] = 'tokenization.parquet'
    with open(config_path, 'w') as file:
        yaml.dump(config, file)

    logger.info("Texto tokenizado e salvo com sucesso.")
    logger.info(f"Arquivo de configuração atualizado em: {config_path}")

if __name__ == "__main__":
    tokenizacao_texto()
