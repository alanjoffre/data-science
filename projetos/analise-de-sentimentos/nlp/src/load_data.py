import logging
import yaml
import dask.dataframe as dd
import pandas as pd

# Carregar arquivo de configuração
config_path = 'D:\\Github\\data-science\\projetos\\analise-de-sentimentos\\nlp\\config\\config.yaml'
with open(config_path, 'r') as file:
    config = yaml.safe_load(file)

# Configuração de Logs
logging.basicConfig(level=config['logs']['level'],
                    format=config['logs']['format'],
                    handlers=[logging.StreamHandler(),
                              logging.FileHandler(config['directories']['logs'] + config['logs']['file'])])
logger = logging.getLogger(__name__)

def carregar_dados():
    logger.info("Iniciando a etapa de carregamento de dados...")

    # Carregar o dataset raw
    raw_data_path = config['directories']['raw_data']
    df = dd.read_parquet(raw_data_path, engine='pyarrow')

    # Renomear colunas
    colunas = ['index', 'id', 'date', 'query', 'username', 'tweet']
    df.columns = colunas

    # Converter Dask DataFrame para Pandas DataFrame para salvar como um único arquivo Parquet
    pandas_df = df.compute()

    # Salvar o dataset processado como um único arquivo Parquet
    processed_data_path = config['directories']['processed_data'] + config['files']['processed_dataset']
    pandas_df.to_parquet(processed_data_path, engine='pyarrow', index=False)
    pandas_df.to_csv(config['directories']['processed_data'] + 'carregamento_de_dados.csv', index=False)

    # Verifique se o arquivo Parquet foi salvo corretamente
    try:
        _ = pd.read_parquet(processed_data_path, engine='pyarrow')
        logger.info("Arquivo Parquet salvo e lido com sucesso.")
    except Exception as e:
        logger.error(f"Erro ao salvar ou ler o arquivo Parquet: {e}")

    # Atualizar o arquivo de configuração
    config['files']['processed_dataset'] = 'carregamento_de_dados.parquet'
    with open(config_path, 'w') as file:
        yaml.dump(config, file)

    logger.info("Dados carregados e salvos com sucesso.")
    logger.info(f"Arquivo de configuração atualizado em: {config_path}")

if __name__ == "__main__":
    carregar_dados()
