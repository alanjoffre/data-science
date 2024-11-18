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
                              logging.FileHandler(config['directories']['logs'] + 'data_cleaning.log')])
logger = logging.getLogger(__name__)

def limpar_dados():
    logger.info("Iniciando a etapa de limpeza de dados...")

    # Carregar o dataset processado da etapa anterior
    processed_data_path = config['directories']['processed_data'] + 'carregamento_de_dados.parquet'
    df = pd.read_parquet(processed_data_path, engine='pyarrow')

    # Converter Pandas DataFrame para Dask DataFrame para processamento em paralelo
    ddf = dd.from_pandas(df, npartitions=4)

    # Remover duplicatas
    ddf = ddf.drop_duplicates()

    # Tratar valores ausentes (por exemplo, remover linhas com valores ausentes)
    ddf = ddf.dropna()

    # Converter Dask DataFrame para Pandas DataFrame
    cleaned_df = ddf.compute()

    # Salvar o dataset limpo como um único arquivo Parquet
    cleaned_data_path = config['directories']['processed_data'] + 'data_cleaning.parquet'
    cleaned_df.to_parquet(cleaned_data_path, engine='pyarrow', index=False)
    cleaned_df.to_csv(config['directories']['processed_data'] + 'data_cleaning.csv', index=False)

    # Verifique se o arquivo Parquet foi salvo corretamente
    try:
        _ = pd.read_parquet(cleaned_data_path, engine='pyarrow')
        logger.info("Arquivo Parquet salvo e lido com sucesso.")
    except Exception as e:
        logger.error(f"Erro ao salvar ou ler o arquivo Parquet: {e}")

    # Atualizar o arquivo de configuração
    config['files']['processed_dataset'] = 'data_cleaning.parquet'
    with open(config_path, 'w') as file:
        yaml.dump(config, file)

    logger.info("Dados limpos e salvos com sucesso.")
    logger.info(f"Arquivo de configuração atualizado em: {config_path}")

if __name__ == "__main__":
    limpar_dados()
