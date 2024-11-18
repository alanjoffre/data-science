import unittest
import yaml
import pandas as pd
import spacy
import os

class TestStemmingLemmatization(unittest.TestCase):

    def setUp(self):
        # Carregar arquivo de configuração
        self.config_path = 'D:\\Github\\data-science\\projetos\\analise-de-sentimentos\\nlp\\config\\config.yaml'
        with open(self.config_path, 'r') as file:
            self.config = yaml.safe_load(file)
        # Carregar o dataset lematizado
        self.lemmatized_data_path = self.config['directories']['processed_data'] + 'stemming_lemmatization.parquet'
        self.nlp = spacy.load('en_core_web_sm', disable=['ner', 'parser'])

    def test_stemming_lemmatization(self):
        # Verificar se o arquivo Parquet existe
        self.assertTrue(os.path.exists(self.lemmatized_data_path), f"Arquivo não encontrado: {self.lemmatized_data_path}")

        # Carregar os dados e verificar se os tokens foram lematizados corretamente
        df = pd.read_parquet(self.lemmatized_data_path, engine='pyarrow')
        def check_lemmatizacao(tokens):
            doc = self.nlp(" ".join(tokens))
            return [token.lemma_ for token in doc] == tokens
        self.assertTrue(df['tweet_tokens'].apply(check_lemmatizacao).all(), "Tokens não foram lematizados corretamente.")

if __name__ == '__main__':
    unittest.main()
