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
                              logging.FileHandler(config['directories']['logs'] + 'stop_words_removal.log')])
logger = logging.getLogger(__name__)

# Carregar o modelo de linguagem do spaCy
nlp = spacy.load('en_core_web_sm', disable=['ner', 'parser'])

def remover_stop_words(texto):
    doc = nlp(texto)
    return " ".join([token.text for token in doc if not token.is_stop])

def remocao_stop_words():
    logger.info("Iniciando a etapa de remoção de stop words...")

    # Carregar o dataset processado da etapa anterior
    processed_data_path = config['directories']['processed_data'] + config['files']['processed_dataset']
    df = dd.read_parquet(processed_data_path, engine='pyarrow')

    # Remover stop words dos textos
    df['tweet_no_stopwords'] = df['tweet'].map_partitions(lambda s: s.apply(remover_stop_words), meta=('tweet_no_stopwords', 'object'))

    # Converter Dask DataFrame para Pandas DataFrame
    cleaned_df = df.compute()

    # Salvar o dataset como um único arquivo Parquet
    stopwords_removed_data_path = config['directories']['processed_data'] + 'stop_words_removal.parquet'
    cleaned_df.to_parquet(stopwords_removed_data_path, engine='pyarrow', index=False)
    cleaned_df.to_csv(config['directories']['processed_data'] + 'stop_words_removal.csv', index=False)

    # Verifique se o arquivo Parquet foi salvo corretamente
    try:
        _ = pd.read_parquet(stopwords_removed_data_path, engine='pyarrow')
        logger.info("Arquivo Parquet salvo e lido com sucesso.")
    except Exception as e:
        logger.error(f"Erro ao salvar ou ler o arquivo Parquet: {e}")

    # Atualizar o arquivo de configuração
    config['files']['processed_dataset'] = 'stop_words_removal.parquet'
    with open(config_path, 'w') as file:
        yaml.dump(config, file)

    logger.info("Stop words removidas e dados salvos com sucesso.")
    logger.info(f"Arquivo de configuração atualizado em: {config_path}")

if __name__ == "__main__":
    remocao_stop_words()
