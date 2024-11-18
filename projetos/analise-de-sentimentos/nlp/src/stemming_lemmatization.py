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
                              logging.FileHandler(config['directories']['logs'] + 'stemming_lemmatization.log')])
logger = logging.getLogger(__name__)

# Carregar o modelo de linguagem do spaCy
nlp = spacy.load('en_core_web_sm', disable=['ner', 'parser'])

def lemmatizar_tokens(tokens):
    doc = nlp(" ".join(tokens))
    return [token.lemma_ for token in doc]

def stemming_lemmatization():
    logger.info("Iniciando a etapa de stemming e lematização...")

    # Carregar o dataset processado da etapa anterior
    processed_data_path = config['directories']['processed_data'] + config['files']['processed_dataset']
    df = dd.read_parquet(processed_data_path, engine='pyarrow')

    # Lematizar os tokens dos tweets
    df['tweet_tokens'] = df['tweet_tokens'].map_partitions(lambda s: s.apply(lemmatizar_tokens), meta=('tweet_tokens', 'object'))

    # Converter Dask DataFrame para Pandas DataFrame
    lemmatized_df = df.compute()

    # Salvar o dataset lematizado como um único arquivo Parquet
    lemmatized_data_path = config['directories']['processed_data'] + 'stemming_lemmatization.parquet'
    lemmatized_df.to_parquet(lemmatized_data_path, engine='pyarrow', index=False)
    lemmatized_df.to_csv(config['directories']['processed_data'] + 'stemming_lemmatization.csv', index=False)

    # Verifique se o arquivo Parquet foi salvo corretamente
    try:
        _ = pd.read_parquet(lemmatized_data_path, engine='pyarrow')
        logger.info("Arquivo Parquet salvo e lido com sucesso.")
    except Exception as e:
        logger.error(f"Erro ao salvar ou ler o arquivo Parquet: {e}")

    # Atualizar o arquivo de configuração
    config['files']['processed_dataset'] = 'stemming_lemmatization.parquet'
    with open(config_path, 'w') as file:
        yaml.dump(config, file)

    logger.info("Stemming e lematização concluídos e dados salvos com sucesso.")
    logger.info(f"Arquivo de configuração atualizado em: {config_path}")

if __name__ == "__main__":
    stemming_lemmatization()
