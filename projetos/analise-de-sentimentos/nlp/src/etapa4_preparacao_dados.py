import dask.dataframe as dd
from sklearn.model_selection import train_test_split
import logging
import yaml

# Carregar arquivo de configuração
config_path = 'D:\\Github\\data-science\\projetos\\analise-de-sentimentos\\nlp\\config\\config.yaml'
with open(config_path, 'r') as file:
    config = yaml.safe_load(file)

# Configuração de Logs
logging.basicConfig(level=config['logs']['level'],
                    format=config['logs']['format'],
                    handlers=[logging.StreamHandler(),
                              logging.FileHandler(config['directories']['logs'] + 'preparacao_dados.log')])
logger = logging.getLogger(__name__)

def dividir_dados(df):
    """Divide o conjunto de dados em treinamento e teste"""
    df_pd = df.compute()
    train_df, test_df = train_test_split(df_pd, test_size=0.2, random_state=42)
    return train_df, test_df

def preparar_dados():
    # Carregar o dataset analisado da etapa anterior
    processed_data_path = config['directories']['processed_data'] + config['files']['processed_dataset']
    logger.info("Carregando dataset analisado do caminho: %s", processed_data_path)

    df = dd.read_parquet(processed_data_path)
    logger.info("Dataset carregado com sucesso!")

    # Divisão de dados
    train_df, test_df = dividir_dados(df)
    logger.info("Conjunto de dados dividido com sucesso!")

    # Salvar os datasets de treinamento e teste
    train_data_parquet = config['directories']['processed_data'] + 'train_data.parquet'
    test_data_parquet = config['directories']['processed_data'] + 'test_data.parquet'
    train_data_csv = config['directories']['processed_data'] + 'train_data.csv'
    test_data_csv = config['directories']['processed_data'] + 'test_data.csv'

    train_df.to_parquet(train_data_parquet, engine='pyarrow', index=False)
    train_df.to_csv(train_data_csv, index=False)
    test_df.to_parquet(test_data_parquet, engine='pyarrow', index=False)
    test_df.to_csv(test_data_csv, index=False)
    logger.info("Conjuntos de dados de treinamento e teste salvos em %s, %s, %s e %s", train_data_parquet, test_data_parquet, train_data_csv, test_data_csv)

    # Atualizar o arquivo de configuração
    config['files']['train_dataset'] = 'train_data.parquet'
    config['files']['test_dataset'] = 'test_data.parquet'
    with open(config_path, 'w') as file:
        yaml.dump(config, file)
    logger.info("Arquivo de configuração atualizado com sucesso.")

if __name__ == "__main__":
    preparar_dados()
