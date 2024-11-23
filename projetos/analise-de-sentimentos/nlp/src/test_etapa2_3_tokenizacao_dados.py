import pytest
import os
import pandas as pd
import yaml
from etapa2_3_tokenizacao_dados import tokenizar_texto, criar_bigramas, tokenizacao_dados

def carregar_config():
    """
    Carrega o arquivo de configuração YAML e retorna como um dicionário.
    """
    config_path = 'D:\\Github\\data-science\\projetos\\analise-de-sentimentos\\nlp\\config\\config.yaml'
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)

    # Atualizar campos específicos para o teste
    config['files']['log_file'] = "etapa2_3_tokenizacao_dados.log"
    config['files']['processed_dataset'] = "etapa2_3_tokenizacao_dados.parquet"
    config['files']['processed_dataset_csv'] = "etapa2_3_tokenizacao_dados.csv"
    config['files']['raw_dataset'] = "asentimentos.parquet"

    # Salvar as mudanças no arquivo YAML
    with open(config_path, 'w') as file:
        yaml.safe_dump(config, file)

    return config


# Carrega a configuração global para os testes
config = carregar_config()

# Diretórios e arquivos para testes
tokenized_data_path = os.path.join(config['directories']['processed_data'], "etapa2_3_tokenizacao_dados.parquet")
tokenized_data_csv_path = os.path.join(config['directories']['processed_data'], "etapa2_3_tokenizacao_dados.csv")


def test_tokenizar_texto():
    """
    Testa a função de tokenização de texto.
    """
    texto = "Hello, world! This is a test."
    tokens = tokenizar_texto(texto)
    assert tokens == ["Hello", "world", "This", "is", "a", "test"]


def test_criar_bigramas():
    """
    Testa a função de criação de bigramas.
    """
    texto = "Hello world this is a test"
    bigrams = criar_bigramas(texto)
    assert bigrams == ["Hello_world", "world_this", "this_is", "is_a", "a_test"]


def test_tokenizacao_dados():
    """
    Testa o processo completo de tokenização e verifica se os arquivos foram salvos corretamente.
    """
    tokenizacao_dados()

    # Verifica se os arquivos foram criados
    assert os.path.exists(tokenized_data_path)
    assert os.path.exists(tokenized_data_csv_path)

    # Carrega o dataset salvo e verifica o conteúdo
    df_parquet = pd.read_parquet(tokenized_data_path)
    df_csv = pd.read_csv(tokenized_data_csv_path)

    assert not df_parquet.empty
    assert not df_csv.empty

    # Verifica se as colunas 'tokens' e 'bigrams' existem
    for df in [df_parquet, df_csv]:
        assert 'tokens' in df.columns
        assert 'bigrams' in df.columns

    # Verifica se o arquivo config.yaml foi atualizado corretamente
    with open('D:\\Github\\data-science\\projetos\\analise-de-sentimentos\\nlp\\config\\config.yaml', 'r') as file:
        config_updated = yaml.safe_load(file)

    assert config_updated['files']['log_file'] == "etapa2_3_tokenizacao_dados.log"
    assert config_updated['files']['processed_dataset'] == "etapa2_3_tokenizacao_dados.parquet"
    assert config_updated['files']['processed_dataset_csv'] == "etapa2_3_tokenizacao_dados.csv"
    assert config_updated['files']['raw_dataset'] == "asentimentos.parquet"


if __name__ == "__main__":
    pytest.main()
