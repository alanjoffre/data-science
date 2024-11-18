import unittest
import yaml
import pandas as pd
import spacy
import os

class TestTokenizacaoTexto(unittest.TestCase):

    def setUp(self):
        # Carregar arquivo de configuração
        self.config_path = 'D:\\Github\\data-science\\projetos\\analise-de-sentimentos\\nlp\\config\\config.yaml'
        with open(self.config_path, 'r') as file:
            self.config = yaml.safe_load(file)
        # Carregar o dataset tokenizado
        self.tokenized_data_path = self.config['directories']['processed_data'] + 'tokenization.parquet'
        self.nlp = spacy.load('en_core_web_sm', disable=['ner', 'parser'])

    def test_tokenizacao_texto(self):
        # Verificar se o arquivo Parquet existe
        self.assertTrue(os.path.exists(self.tokenized_data_path), f"Arquivo não encontrado: {self.tokenized_data_path}")

        # Carregar os dados e verificar se o texto foi tokenizado corretamente
        df = pd.read_parquet(self.tokenized_data_path, engine='pyarrow')
        def check_tokenizacao(texto, tokens):
            doc = self.nlp(texto)
            return [token.text for token in doc] == tokens
        self.assertTrue(df.apply(lambda row: check_tokenizacao(row['tweet'], row['tweet_tokens']), axis=1).all(), "Texto não tokenizado corretamente.")

if __name__ == '__main__':
    unittest.main()
