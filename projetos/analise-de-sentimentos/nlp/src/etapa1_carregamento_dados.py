import dask.dataframe as dd
import logging
import yaml
import os
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
                              logging.FileHandler(os.path.join(config['directories']['logs'], config['files']['log_file']))])
logger = logging.getLogger(__name__)

def atualizar_config(config, chave, valor):
    config[chave] = valor
    with open(config_path, 'w') as file:
        yaml.safe_dump(config, file)
    logger.info("Arquivo config.yaml atualizado com sucesso.")

def carregar_dados():
    try:
        logger.info("Carregando o dataset bruto.")
        raw_data_path = os.path.join(config['directories']['raw_data'], config['files']['raw_dataset'])
        df = dd.read_parquet(raw_data_path)
        logger.info("Dataset carregado com sucesso.")
        
        logger.info("Renomeando colunas.")
        colunas_novas = ["index", "id", "date", "query", "username", "tweet"]
        df.columns = colunas_novas
        
        logger.info("Salvando dataset processado em formato Parquet e CSV.")
        processed_data_path = os.path.join(config['directories']['processed_data'], config['files']['processed_dataset'])
        processed_data_csv_path = os.path.join(config['directories']['processed_data'], config['files']['processed_dataset_csv'])
        
        # Salvar como arquivo Parquet único
        df.to_csv(processed_data_csv_path, single_file=True)
        table = pa.Table.from_pandas(df.compute())
        pq.write_table(table, processed_data_path)
        
        logger.info("Dataset salvo com sucesso.")
        
        
        # Atualizar config.yaml para a próxima etapa
        atualizar_config(config, 'files', {'log_file': "etapa1_carregamento_dados.log",
                                           'processed_dataset': "processed_dataset.parquet",
                                           'processed_dataset_csv': "processed_dataset.csv",
                                           'raw_dataset': "asentimentos.parquet"})
               
    except Exception as e:
        logger.error("Erro ao carregar os dados: %s", e)
        raise

if __name__ == "__main__":
    carregar_dados()
