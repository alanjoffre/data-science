import pandas as pd
import yaml
import logging

# Carregar arquivo de configuração
config_path = 'D:\\Github\\data-science\\projetos\\analise-de-sentimentos\\nlp\\config\\config.yaml'
with open(config_path, 'r') as file:
    config = yaml.safe_load(file)

# Configuração de Logs
logging.basicConfig(level=logging.INFO)

def test_preparar_dados():
    train_data_path = config['directories']['processed_data'] + 'train_data.parquet'
    test_data_path = config['directories']['processed_data'] + 'test_data.parquet'
    try:
        train_df = pd.read_parquet(train_data_path)
        test_df = pd.read_parquet(test_data_path)
        assert 'tweet' in train_df.columns
        assert 'tweet' in test_df.columns
        logging.info("Teste de preparação de dados: SUCESSO!")
    except Exception as e:
        logging.error("Teste de preparação de dados: FALHA! Erro: %s", e)

if __name__ == "__main__":
    test_preparar_dados()
