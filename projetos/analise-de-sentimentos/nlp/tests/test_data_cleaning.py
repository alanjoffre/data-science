import unittest
import yaml
import pandas as pd
import os

class TestLimpezaDados(unittest.TestCase):

    def setUp(self):
        # Carregar arquivo de configuração
        self.config_path = 'D:\\Github\\data-science\\projetos\\analise-de-sentimentos\\nlp\\config\\config.yaml'
        with open(self.config_path, 'r') as file:
            self.config = yaml.safe_load(file)
        # Carregar o dataset limpo
        self.cleaned_data_path = self.config['directories']['processed_data'] + 'data_cleaning.parquet'

    def test_limpeza_dados(self):
        # Verificar se o arquivo Parquet existe
        self.assertTrue(os.path.exists(self.cleaned_data_path), f"Arquivo não encontrado: {self.cleaned_data_path}")

        # Carregar os dados e verificar se as duplicatas e valores ausentes foram removidos
        df = pd.read_parquet(self.cleaned_data_path, engine='pyarrow')
        self.assertFalse(df.duplicated().any(), "Ainda há duplicatas nos dados.")
        self.assertFalse(df.isnull().values.any(), "Ainda há valores ausentes nos dados.")

if __name__ == '__main__':
    unittest.main()
