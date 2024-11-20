import pandas as pd
import yaml
import logging

# Carregar arquivo de configuração
config_path = 'D:\\Github\\data-science\\projetos\\analise-de-sentimentos\\nlp\\config\\config.yaml'
with open(config_path, 'r') as file:
    config = yaml.safe_load(file)

# Configuração de Logs
logging.basicConfig(level=logging.INFO)

def test_analise_exploratoria():
    processed_data_path = config['directories']['processed_data'] + 'etapa3_analise_exploratoria.parquet'
    try:
        df = pd.read_parquet(processed_data_path)
        assert 'tweet' in df.columns
        assert df['tweet'].str.len().min() > 0
        logging.info("Teste de análise exploratória de dados: SUCESSO!")
    except Exception as e:
        logging.error("Teste de análise exploratória de dados: FALHA! Erro: %s", e)

if __name__ == "__main__":
    test_analise_exploratoria()
