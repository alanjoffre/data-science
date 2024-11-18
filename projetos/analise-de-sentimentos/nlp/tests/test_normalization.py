import unittest
import yaml
import pandas as pd
import re
import os

class TestNormalizacaoTexto(unittest.TestCase):

    def setUp(self):
        # Carregar arquivo de configuração
        self.config_path = 'D:\\Github\\data-science\\projetos\\analise-de-sentimentos\\nlp\\config\\config.yaml'
        with open(self.config_path, 'r') as file:
            self.config = yaml.safe_load(file)
        # Carregar o dataset normalizado
        self.normalized_data_path = self.config['directories']['processed_data'] + 'text_normalization.parquet'

    def test_normalizacao_texto(self):
        # Verificar se o arquivo Parquet existe
        self.assertTrue(os.path.exists(self.normalized_data_path), f"Arquivo não encontrado: {self.normalized_data_path}")

        # Carregar os dados e verificar se o texto foi normalizado corretamente
        df = pd.read_parquet(self.normalized_data_path, engine='pyarrow')
        def check_normalizacao(texto):
            return re.match(r'^[a-z\s]*$', texto) is not None
        self.assertTrue(df['tweet'].apply(check_normalizacao).all(), "Texto não normalizado corretamente.")

if __name__ == '__main__':
    unittest.main()
