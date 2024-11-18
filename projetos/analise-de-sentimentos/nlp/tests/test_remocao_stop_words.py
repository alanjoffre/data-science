import unittest
import yaml
import pandas as pd
import spacy
import os

class TestRemocaoStopWords(unittest.TestCase):

    def setUp(self):
        # Carregar arquivo de configuração
        self.config_path = 'D:\\Github\\data-science\\projetos\\analise-de-sentimentos\\nlp\\config\\config.yaml'
        with open(self.config_path, 'r') as file:
            self.config = yaml.safe_load(file)
        # Carregar o dataset com stop words removidas
        self.stopwords_removed_data_path = self.config['directories']['processed_data'] + 'stop_words_removal.parquet'
        self.nlp = spacy.load('en_core_web_sm', disable=['ner', 'parser'])

    def test_remocao_stop_words(self):
        # Verificar se o arquivo Parquet existe
        self.assertTrue(os.path.exists(self.stopwords_removed_data_path), f"Arquivo não encontrado: {self.stopwords_removed_data_path}")

        # Carregar os dados e verificar se as stop words foram removidas corretamente
        df = pd.read_parquet(self.stopwords_removed_data_path, engine='pyarrow')
        def check_stop_words(texto):
            doc = self.nlp(texto)
            return not any(token.is_stop for token in doc)
        self.assertTrue(df['tweet_no_stopwords'].apply(check_stop_words).all(), "Stop words não foram removidas corretamente.")

if __name__ == '__main__':
    unittest.main()
