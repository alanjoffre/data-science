import dask.dataframe as dd
import yaml
import logging

# Carregar arquivo de configuração
config_path = 'D:\\Github\\data-science\\projetos\\analise-de-sentimentos\\nlp\\config\\config.yaml'
with open(config_path, 'r') as file:
    config = yaml.safe_load(file)

# Configuração de Logs
logging.basicConfig(level=logging.INFO)

def test_carregar_dados():
    processed_data_path = config['directories']['processed_data'] + 'etapa1_carregamento_dados.parquet'
    try:
        df = dd.read_parquet(processed_data_path)
        assert df.columns.tolist() == ['index', 'id', 'date', 'query', 'username', 'tweet']
        logging.info("Teste de carregamento de dados: SUCESSO!")
    except Exception as e:
        logging.error("Teste de carregamento de dados: FALHA! Erro: %s", e)

if __name__ == "__main__":
    test_carregar_dados()
