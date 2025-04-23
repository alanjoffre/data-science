import dask.dataframe as dd
import yaml
import logging

# Configurar o logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Carregar arquivo de configuração
config_path = 'D:\\Github\\data-science\\projetos\\analise-de-sentimentos\\nlp\\config\\config.yaml'
with open(config_path, 'r') as file:
    config = yaml.safe_load(file)

def criar_amostra():
    # Caminho do arquivo original
    raw_data_path = 'D:\\Github\\data-science\\projetos\\analise-de-sentimentos\\nlp\\data\\raw\\asentimentos.parquet'
    logger.info("Carregando dataset original do caminho: %s", raw_data_path)

    # Carregar o dataset bruto
    df = dd.read_parquet(raw_data_path)
    logger.info("Dataset carregado com sucesso!")

    # Selecionar as primeiras 200000 linhas
    logger.info("Selecionando as primeiras 200000 linhas do dataset.")
    amostra_df = df.head(200000)  # 'head' carrega apenas as primeiras linhas no formato pandas

    # Caminho para salvar a amostra
    amostra_path = config['directories']['raw_data'] + 'amostra.parquet'

    # Salvar a amostra em Parquet
    logger.info("Salvando a amostra no caminho: %s", amostra_path)
    amostra_df.to_parquet(amostra_path, engine='pyarrow', index=False)
    logger.info("Amostra salva com sucesso!")

    # Exibir as primeiras 5 linhas para validação
    print(amostra_df.head())

if __name__ == "__main__":
    criar_amostra()
