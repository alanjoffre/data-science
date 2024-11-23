import pytest
import pandas as pd
import dask.dataframe as dd
import os
import yaml
from etapa2_6_vetorizacao_texto import (
    carregar_dados, preprocessar_dados, aplicar_tfidf, salvar_dados, atualizar_config, vetorizacao_texto
)

@pytest.fixture
def sample_data():
    data = {
        'tokens_lemmatized': [
            ['word1', 'word2', 'word3'],
            ['word1', 'word2'],
            ['word1'],
            [],
            ['word3', 'word4']
        ]
    }
    df = pd.DataFrame(data)
    return dd.from_pandas(df, npartitions=1)

def test_carregar_dados(monkeypatch, sample_data):
    def mock_read_parquet(*args, **kwargs):
        return sample_data
    monkeypatch.setattr(dd, 'read_parquet', mock_read_parquet)
    
    df = carregar_dados()
    assert not df.compute().empty
    assert len(df) == 5

def test_preprocessar_dados(sample_data):
    df = preprocessar_dados(sample_data)
    df = df.compute()
    
    assert len(df) == 4  # A linha vazia deve ser removida
    assert all(isinstance(x, str) for x in df['tokens_lemmatized'])

def test_aplicar_tfidf(sample_data):
    df = preprocessar_dados(sample_data)
    df_tfidf = aplicar_tfidf(df)
    
    assert df_tfidf is not None
    assert 'word1' in df_tfidf.columns
    assert 'word2' in df_tfidf.columns

def test_salvar_dados(tmpdir, sample_data):
    df = preprocessar_dados(sample_data)
    df_tfidf = aplicar_tfidf(df)
    
    salvar_dados(df_tfidf, str(tmpdir))
    
    parquet_path = os.path.join(tmpdir, 'etapa2_6_vetorizacao_texto.parquet')
    csv_path = os.path.join(tmpdir, 'etapa2_6_vetorizacao_texto.csv')
    
    assert os.path.exists(parquet_path)
    assert os.path.exists(csv_path)
    
    df_parquet = pd.read_parquet(parquet_path)
    df_csv = pd.read_csv(csv_path)
    
    assert not df_parquet.empty
    assert not df_csv.empty

def test_atualizar_config(monkeypatch, tmpdir):
    # Caminho do arquivo de configuração
    config_path = 'D:\\Github\\data-science\\projetos\\analise-de-sentimentos\\nlp\\config\\config.yaml'

    # Carregar o arquivo de configuração e atualizar caminhos para o diretório temporário
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)

    # Atualizar os campos específicos no arquivo de configuração
    config['files']['log_file'] = 'etapa2_6_vetorizacao_texto.log'
    config['files']['processed_dataset'] = 'etapa2_6_vetorizacao_texto.parquet'
    config['files']['processed_dataset_csv'] = 'etapa2_6_vetorizacao_texto.csv'
    config['files']['raw_dataset'] = 'asentimentos.parquet'

    # Salvar as atualizações no arquivo de configuração
    with open(config_path, 'w') as file:
        yaml.safe_dump(config, file)

    # Mock da função carregar_dados para usar dados de amostra
    def mock_carregar_dados():
        data = {
            'tokens_lemmatized': [
                ['word1', 'word2', 'word3'],
                ['word1', 'word2'],
                ['word1'],
                [],
                ['word3', 'word4']
            ]
        }
        df = pd.DataFrame(data)
        return dd.from_pandas(df, npartitions=1)
    
    monkeypatch.setattr('etapa2_6_vetorizacao_texto.carregar_dados', mock_carregar_dados)
    
    # Mock da função atualizar_config para usar caminho temporário durante o teste
    def mock_atualizar_config(config, chave, valor, config_path=config_path):
        config[chave] = valor
        with open(config_path, 'w') as file:
            yaml.safe_dump(config, file)
    monkeypatch.setattr('etapa2_6_vetorizacao_texto.atualizar_config', mock_atualizar_config)

    # Executar o script de vetorização
    vetorizacao_texto()

    # Verificar se o config.yaml foi atualizado corretamente
    with open(config_path, 'r') as file:
        config_updated = yaml.safe_load(file)

    assert config_updated['files']['log_file'] == 'etapa2_6_vetorizacao_texto.log'
    assert config_updated['files']['processed_dataset'] == 'etapa2_6_vetorizacao_texto.parquet'
    assert config_updated['files']['processed_dataset_csv'] == 'etapa2_6_vetorizacao_texto.csv'
    assert config_updated['files']['raw_dataset'] == 'asentimentos.parquet'

if __name__ == "__main__":
    pytest.main()
