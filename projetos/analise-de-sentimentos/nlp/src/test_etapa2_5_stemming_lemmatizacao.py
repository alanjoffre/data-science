import unittest
import dask.dataframe as dd
import yaml
import os
import spacy
from nltk.stem import SnowballStemmer

class TestStemmingLemmatizacao(unittest.TestCase):
    def setUp(self):
        config_path = 'D:\\Github\\data-science\\projetos\\analise-de-sentimentos\\nlp\\config\\config.yaml'
        with open(config_path, 'r') as file:
            self.config = yaml.safe_load(file)
            
        # Atualizar campos específicos para o teste
        self.config['files']['log_file'] = "etapa2_5_stemming_lemmatizacao.log"
        self.config['files']['processed_dataset'] = "etapa2_5_stemming_lemmatizacao.parquet"
        self.config['files']['processed_dataset_csv'] = "etapa2_5_stemming_lemmatizacao.csv"
        self.config['files']['raw_dataset'] = "asentimentos.parquet"
        with open('D:\\Github\\data-science\\projetos\\analise-de-sentimentos\\nlp\\config\\config.yaml', 'w') as file:
            yaml.safe_dump(self.config, file)
            
        # Carregar modelo de linguagem spaCy
        self.nlp = spacy.load("en_core_web_sm")
        # Inicializar o stemmer do NLTK
        self.stemmer = SnowballStemmer(language='english')
        
    def test_stemming_lemmatizacao(self):
        stopwords_removed_data_path = os.path.join(self.config['directories']['processed_data'], self.config['files']['processed_dataset'])
        stemmed_lemmatized_data_path = os.path.join(self.config['directories']['processed_data'], "etapa2_5_stemming_lemmatizacao.parquet")
        
        # Testar se o dataset com stopwords removidas foi carregado
        df = dd.read_parquet(stopwords_removed_data_path)
        self.assertGreater(len(df.index), 0, "O dataset com stopwords removidas não foi carregado corretamente.")
        
        # Testar se o dataset com stemming e lematização foi salvo
        df_stemmed_lemmatized = dd.read_parquet(stemmed_lemmatized_data_path)
        self.assertGreater(len(df_stemmed_lemmatized.index), 0, "O dataset com stemming e lematização não foi salvo corretamente.")
        
        # Verificar se o stemming foi aplicado corretamente
        tokens_stemmed = df_stemmed_lemmatized['tokens_stemmed'].compute()
        for token_list in tokens_stemmed:
            stemmed_tokens = [self.stemmer.stem(token) for token in token_list]
            self.assertEqual([token.lower() for token in token_list], [token.lower() for token in stemmed_tokens], "Stemming não foi aplicado corretamente.")
        
        # Verificar se a lematização foi aplicada corretamente
        tokens_lemmatized = df_stemmed_lemmatized['tokens_lemmatized'].compute()
        for token_list in tokens_lemmatized:
            doc = self.nlp(" ".join(token_list))
            lemmatized_tokens = [token.lemma_.lower().strip() for token in doc if token.lemma_.strip()]
            self.assertEqual([token.lower().strip() for token in token_list if token.strip()], lemmatized_tokens, "Lematização não foi aplicada corretamente.")
        
        # Testar se o config.yaml foi atualizado
        with open('D:\\Github\\data-science\\projetos\\analise-de-sentimentos\\nlp\\config\\config.yaml', 'r') as file:
            config_updated = yaml.safe_load(file)
        
        self.assertEqual(config_updated['files']['log_file'], "etapa2_5_stemming_lemmatizacao.log", "O arquivo config.yaml não foi atualizado corretamente.")
        self.assertEqual(config_updated['files']['processed_dataset'], "etapa2_5_stemming_lemmatizacao.parquet", "O arquivo config.yaml não foi atualizado corretamente.")
        self.assertEqual(config_updated['files']['processed_dataset_csv'], "etapa2_5_stemming_lemmatizacao.csv", "O arquivo config.yaml não foi atualizado corretamente.")
        self.assertEqual(config_updated['files']['raw_dataset'], "asentimentos.parquet", "O arquivo config.yaml não foi atualizado corretamente.")    

if __name__ == "__main__":
    unittest.main()
