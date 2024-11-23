import pytest
import pandas as pd
import os

@pytest.fixture
def sample_data():
    data = {
        'col1': [1, 2, 3, 4, 5],
        'col2': [5, 4, 3, 2, 1]
    }
    df = pd.DataFrame(data)
    return df

def test_estatisticas_descritivas(sample_data, tmpdir):
    csv_path = os.path.join(tmpdir, 'estatisticas_descritivas.csv')
    parquet_path = os.path.join(tmpdir, 'estatisticas_descritivas.parquet')

    estatisticas_descritivas = sample_data.describe()
    estatisticas_descritivas.to_csv(csv_path)
    estatisticas_descritivas.to_parquet(parquet_path)

    assert os.path.exists(csv_path)
    assert os.path.exists(parquet_path)

    df_loaded_csv = pd.read_csv(csv_path)
    df_loaded_parquet = pd.read_parquet(parquet_path)

    assert not df_loaded_csv.empty
    assert not df_loaded_parquet.empty
    assert list(df_loaded_csv.columns) == ['Unnamed: 0', 'col1', 'col2']
    assert list(df_loaded_parquet.columns) == ['col1', 'col2']

print("Teste de Análise Estatística concluído com sucesso.")
