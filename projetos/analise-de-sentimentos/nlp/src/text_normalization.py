import logging
import yaml
import dask.dataframe as dd
import pandas as pd
import re

# Carregar arquivo de configuração
config_path = 'D:\\Github\\data-science\\projetos\\analise-de-sentimentos\\nlp\\config\\config.yaml'
with open(config_path, 'r') as file:
    config = yaml.safe_load(file)

# Configuração de Logs
logging.basicConfig(level=config['logs']['level'],
                    format=config['logs']['format'],
                    handlers=[logging.StreamHandler(),
                              logging.FileHandler(config['directories']['logs'] + 'text_normalization.log')])
logger = logging.getLogger(__name__)

def normalizar_texto(texto):
    # Verificar se o texto é NaN
    if pd.isna(texto):
        return ""
    # Converter para minúsculas
    texto = texto.lower()
    # Remover pontuações, números e caracteres especiais
    texto = re.sub(r'[^a-zA-Z\s]', '', texto)
    return texto

def normalizacao_texto():
    logger.info("Iniciando a etapa de normalização de texto...")

    # Carregar o dataset processado da etapa anterior
    processed_data_path = config['directories']['processed_data'] + config['files']['processed_dataset']
    df = dd.read_parquet(processed_data_path, engine='pyarrow')

    # Normalizar o texto dos tweets
    df['tweet'] = df['tweet'].map_partitions(lambda s: s.apply(normalizar_texto), meta=('tweet', 'object'))

    # Converter Dask DataFrame para Pandas DataFrame
    normalized_df = df.compute()

    # Salvar o dataset normalizado como um único arquivo Parquet
    normalized_data_path = config['directories']['processed_data'] + 'text_normalization.parquet'
    normalized_df.to_parquet(normalized_data_path, engine='pyarrow', index=False)
    normalized_df.to_csv(config['directories']['processed_data'] + 'text_normalization.csv', index=False)

    # Verifique se o arquivo Parquet foi salvo corretamente
    try:
        _ = pd.read_parquet(normalized_data_path, engine='pyarrow')
        logger.info("Arquivo Parquet salvo e lido com sucesso.")
    except Exception as e:
        logger.error(f"Erro ao salvar ou ler o arquivo Parquet: {e}")

    # Atualizar o arquivo de configuração
    config['files']['processed_dataset'] = 'text_normalization.parquet'
    with open(config_path, 'w') as file:
        yaml.dump(config, file)

    logger.info("Texto normalizado e salvo com sucesso.")
    logger.info(f"Arquivo de configuração atualizado em: {config_path}")

if __name__ == "__main__":
    normalizacao_texto()
