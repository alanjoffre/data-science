import unittest
import pandas as pd
import yaml
import os

class TestRemocaoStopwords(unittest.TestCase):
    def setUp(self):
        config_path = 'D:\\Github\\data-science\\projetos\\analise-de-sentimentos\\nlp\\config\\config.yaml'
        with open(config_path, 'r') as file:
            self.config = yaml.safe_load(file)
            
        # Atualizar campos específicos para o teste
        self.config['files']['log_file'] = "etapa2_4_remocao_stopwords.log"
        self.config['files']['processed_dataset'] = "etapa2_4_remocao_stopwords.parquet"
        self.config['files']['processed_dataset_csv'] = "etapa2_4_remocao_stopwords.csv"
        self.config['files']['raw_dataset'] = "asentimentos.parquet"
        with open('D:\\Github\\data-science\\projetos\\analise-de-sentimentos\\nlp\\config\\config.yaml', 'w') as file:
            yaml.safe_dump(self.config, file)
        
    def test_remocao_stopwords(self):
        tokenized_data_path = os.path.join(self.config['directories']['processed_data'], self.config['files']['processed_dataset'])
        stopwords_removed_data_path = os.path.join(self.config['directories']['processed_data'], "etapa2_4_remocao_stopwords.parquet")
        
        # Testar se o dataset tokenizado foi carregado
        df = pd.read_parquet(tokenized_data_path)
        self.assertGreater(len(df.index), 0, "O dataset tokenizado não foi carregado corretamente.")
        
        # Testar se o dataset com stopwords removidas foi salvo
        df_stopwords_removed = pd.read_parquet(stopwords_removed_data_path)
        self.assertGreater(len(df_stopwords_removed.index), 0, "O dataset com stopwords removidas não foi salvo corretamente.")
        
        # Verificar se as stopwords foram removidas corretamente
        tokens = df_stopwords_removed['tokens_sem_stopwords'].head()
        stopwords = set([
            'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your', 'yours', 'yourself', 'yourselves',
            'he', 'him', 'his', 'himself', 'she', 'her', 'hers', 'herself', 'it', 'its', 'itself', 'they', 'them', 'their',
            'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', 'these', 'those', 'am', 'is', 'are',
            'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an',
            'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about',
            'against', 'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up',
            'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when',
            'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor',
            'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', 'should', 'now'
        ])
        for token_list in tokens:
            token_list = token_list.split()  # Converter de string para lista
            self.assertTrue(all(token.lower() not in stopwords for token in token_list), "Stopwords não foram removidas corretamente.")
        
        # Testar se o config.yaml foi atualizado
        with open('D:\\Github\\data-science\\projetos\\analise-de-sentimentos\\nlp\\config\\config.yaml', 'r') as file:
            config_updated = yaml.safe_load(file)
        
        self.assertEqual(config_updated['files']['log_file'], "etapa2_4_remocao_stopwords.log", "O arquivo config.yaml não foi atualizado corretamente.")
        self.assertEqual(config_updated['files']['processed_dataset'], "etapa2_4_remocao_stopwords.parquet", "O arquivo config.yaml não foi atualizado corretamente.")
        self.assertEqual(config_updated['files']['processed_dataset_csv'], "etapa2_4_remocao_stopwords.csv", "O arquivo config.yaml não foi atualizado corretamente.")
        self.assertEqual(config_updated['files']['raw_dataset'], "asentimentos.parquet", "O arquivo config.yaml não foi atualizado corretamente.")    

if __name__ == "__main__":
    unittest.main()
