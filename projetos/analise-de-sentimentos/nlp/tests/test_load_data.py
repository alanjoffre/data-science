import unittest
import yaml
import pandas as pd
import os

class TestCarregamentoDados(unittest.TestCase):

    def setUp(self):
        # Carregar arquivo de configuração
        self.config_path = 'D:\\Github\\data-science\\projetos\\analise-de-sentimentos\\nlp\\config\\config.yaml'
        with open(self.config_path, 'r') as file:
            self.config = yaml.safe_load(file)
        # Carregar o dataset processado
        self.processed_data_path = self.config['directories']['processed_data'] + 'carregamento_de_dados.parquet'

    def test_carregamento_dados(self):
        # Verificar se o arquivo Parquet existe
        self.assertTrue(os.path.exists(self.processed_data_path), f"Arquivo não encontrado: {self.processed_data_path}")

        # Carregar os dados e verificar se as colunas estão corretas
        df = pd.read_parquet(self.processed_data_path, engine='pyarrow')
        colunas = ['index', 'id', 'date', 'query', 'username', 'tweet']
        self.assertEqual(list(df.columns), colunas, f"Colunas incorretas: {list(df.columns)}")

if __name__ == '__main__':
    unittest.main()
