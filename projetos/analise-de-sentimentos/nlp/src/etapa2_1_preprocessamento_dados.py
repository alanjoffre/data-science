import dask.dataframe as dd
import logging
import yaml
import os
import re
import emoji
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
                              logging.FileHandler(os.path.join(config['directories']['logs'], "etapa2_1_preprocessamento_dados.log"))])
logger = logging.getLogger(__name__)

def atualizar_config(config, chave, valor):
    config[chave] = valor
    with open(config_path, 'w') as file:
        yaml.safe_dump(config, file)
    logger.info("Arquivo config.yaml atualizado com sucesso.")

def limpar_texto(texto):
    texto = re.sub(r"http\S+", "", texto)  # Remover links
    texto = re.sub(r"@\w+", "", texto)  # Remover menções
    texto = re.sub(r"#\w+", "", texto)  # Remover hashtags
    texto = emoji.replace_emoji(texto, replace='')  # Remover emojis
    texto = re.sub(r"\d+", "", texto)  # Remover números
    texto = re.sub(r"[^\w\s]", "", texto)  # Remover caracteres especiais
    texto = texto.strip()  # Remover espaços extras
    return texto

def preprocessar_dados():
    try:
        logger.info("Carregando o dataset processado.")
        processed_data_path = os.path.join(config['directories']['processed_data'], config['files']['processed_dataset'])
        df = dd.read_parquet(processed_data_path)
        logger.info("Dataset carregado com sucesso.")
        
        logger.info("Limpando os textos.")
        df['tweet'] = df['tweet'].map(limpar_texto)
        
        logger.info("Salvando dataset pré-processado em formato Parquet e CSV.")
        preprocessed_data_path = os.path.join(config['directories']['processed_data'], "etapa2_1_preprocessamento_dados.parquet")
        preprocessed_data_csv_path = os.path.join(config['directories']['processed_data'], "etapa2_1_preprocessamento_dados.csv")
        
        df.to_csv(preprocessed_data_csv_path, single_file=True)
        table = pa.Table.from_pandas(df.compute())
        pq.write_table(table, preprocessed_data_path)
        
        logger.info("Dataset salvo com sucesso.")
        
        # Atualizar config.yaml para a próxima etapa
        atualizar_config(config, 'files', {'log_file': "etapa2_1_preprocessamento_dados.log",
                                           'processed_dataset': "etapa2_1_preprocessamento_dados.parquet",
                                           'processed_dataset_csv': "etapa2_1_preprocessamento_dados.csv",
                                           'raw_dataset': "asentimentos.parquet"})
              
                        
    except Exception as e:
        logger.error("Erro ao pré-processar os dados: %s", e)
        raise

if __name__ == "__main__":
    preprocessar_dados()
