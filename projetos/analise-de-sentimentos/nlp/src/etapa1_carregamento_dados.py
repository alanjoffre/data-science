import dask.dataframe as dd
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
                              logging.FileHandler(config['directories']['logs'] + 'carregamento_dados.log')])
logger = logging.getLogger(__name__)

def carregar_dados():
    # Carregar o dataset bruto
    raw_data_path = config['directories']['raw_data'] + config['files']['raw_dataset']
    logger.info("Carregando dataset do caminho: %s", raw_data_path)

    df = dd.read_parquet(raw_data_path)
    logger.info("Dataset carregado com sucesso!")

    # Renomear as colunas corretamente
    df.columns = ['index', 'id', 'date', 'query', 'username', 'tweet']
    logger.info("Colunas renomeadas com sucesso!")

    # Converter DataFrame Dask em pandas para salvar e imprimir
    df_pd = df.compute()

    # Salvar o dataset processado diretamente nos arquivos CSV e Parquet
    processed_data_parquet = config['directories']['processed_data'] + 'etapa1_carregamento_dados.parquet'
    processed_data_csv = config['directories']['processed_data'] + 'etapa1_carregamento_dados.csv'
    df_pd.to_parquet(processed_data_parquet, engine='pyarrow', index=False)
    df_pd.to_csv(processed_data_csv, index=False)
    logger.info("Dataset processado salvo em %s e %s", processed_data_parquet, processed_data_csv)

    # Imprimir as primeiras 5 linhas para verificação
    print(df_pd.head())

    # Atualizar o arquivo de configuração
    config['files']['processed_dataset'] = 'etapa1_carregamento_dados.parquet'
    with open(config_path, 'w') as file:
        yaml.dump(config, file)
    logger.info("Arquivo de configuração atualizado com sucesso.")

if __name__ == "__main__":
    carregar_dados()
