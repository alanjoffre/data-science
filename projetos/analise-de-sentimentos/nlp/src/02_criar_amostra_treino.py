import pandas as pd
import dask.dataframe as dd
import yaml
import logging
import os

# Configurar o logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Carregar arquivo de configuração
config_path = 'D:\\Github\\data-science\\projetos\\analise-de-sentimentos\\nlp\\config\\config.yaml'
with open(config_path, 'r') as file:
    config = yaml.safe_load(file)

def criar_amostra():
    # Caminho do arquivo original
    raw_data_path = 'D:\\Github\\data-science\\projetos\\analise-de-sentimentos\\nlp\\data\\processed\\final_data.parquet'
    logger.info("Carregando dataset original do caminho: %s", raw_data_path)

    # Carregar o dataset bruto
    df = dd.read_parquet(raw_data_path)
    logger.info("Dataset carregado com sucesso!")

    # Selecionar 600 linhas de cada sentimento: positivo, neutro e negativo
    positivo_df = df[df['sentimento'] == 'positivo'].sample(frac=1).head(15000)
    neutro_df = df[df['sentimento'] == 'neutro'].sample(frac=1).head(15000)
    negativo_df = df[df['sentimento'] == 'negativo'].sample(frac=1).head(15000)

    amostra_df = dd.concat([positivo_df, neutro_df, negativo_df]).compute()

    # Caminho para salvar a amostra
    sample_parquet_path = 'D:\\Github\\data-science\\projetos\\analise-de-sentimentos\\nlp\\data\\processed\\amostra_treino.parquet'
    sample_csv_path = 'D:\\Github\\data-science\\projetos\\analise-de-sentimentos\\nlp\\data\\processed\\amostra_treino.csv'
    
    # Verificar se o arquivo já existe
    if os.path.exists(sample_parquet_path):
        os.remove(sample_parquet_path)
    
    amostra_df.to_parquet(sample_parquet_path, index=False)
    amostra_df.to_csv(sample_csv_path, index=False)
    logger.info("Amostra salva com sucesso!")

    # Exibir as primeiras 5 linhas para validação
    print(amostra_df.head())

if __name__ == "__main__":
    criar_amostra()
