import unittest
import dask.dataframe as dd
import yaml
import os

class TestNormalizacaoDados(unittest.TestCase):
    def setUp(self):
        config_path = 'D:\\Github\\data-science\\projetos\\analise-de-sentimentos\\nlp\\config\\config.yaml'
        with open(config_path, 'r') as file:
            self.config = yaml.safe_load(file)
            
        # Atualizar campos específicos para o teste
        self.config['files']['log_file'] = "etapa2_2_normalizacao_dados.log"
        self.config['files']['processed_dataset'] = "etapa2_2_normalizacao_dados.parquet"
        self.config['files']['processed_dataset_csv'] = "etapa2_2_normalizacao_dados.csv"
        self.config['files']['raw_dataset'] = "asentimentos.parquet"
        with open('D:\\Github\\data-science\\projetos\\analise-de-sentimentos\\nlp\\config\\config.yaml', 'w') as file:
            yaml.safe_dump(self.config, file)
        
    def test_normalizar_dados(self):
        preprocessed_data_path = os.path.join(self.config['directories']['processed_data'], self.config['files']['processed_dataset'])
        normalized_data_path = os.path.join(self.config['directories']['processed_data'], "etapa2_2_normalizacao_dados.parquet")
        
        # Testar se o dataset pré-processado foi carregado
        df = dd.read_parquet(preprocessed_data_path)
        self.assertGreater(len(df.index), 0, "O dataset pré-processado não foi carregado corretamente.")
        
        # Testar se o dataset normalizado foi salvo
        df_normalized = dd.read_parquet(normalized_data_path)
        self.assertGreater(len(df_normalized.index), 0, "O dataset normalizado não foi salvo corretamente.")
        
        # Verificar se os textos foram normalizados corretamente
        tweets = df_normalized['tweet'].head()
        for tweet in tweets:
            self.assertNotIn("n't", tweet, "Não expandiu as contrações corretamente.")
            self.assertNotIn("'m", tweet, "Não expandiu as contrações corretamente.")
            self.assertNotIn("'ve", tweet, "Não expandiu as contrações corretamente.")
            self.assertNotIn("'ll", tweet, "Não expandiu as contrações corretamente.")
        
        # Testar se o config.yaml foi atualizado
        with open('D:\\Github\\data-science\\projetos\\analise-de-sentimentos\\nlp\\config\\config.yaml', 'r') as file:
            config_updated = yaml.safe_load(file)
        
        self.assertEqual(config_updated['files']['log_file'], "etapa2_2_normalizacao_dados.log", "O arquivo config.yaml não foi atualizado corretamente.")
        self.assertEqual(config_updated['files']['processed_dataset'], "etapa2_2_normalizacao_dados.parquet", "O arquivo config.yaml não foi atualizado corretamente.")
        self.assertEqual(config_updated['files']['processed_dataset_csv'], "etapa2_2_normalizacao_dados.csv", "O arquivo config.yaml não foi atualizado corretamente.")
        self.assertEqual(config_updated['files']['raw_dataset'], "asentimentos.parquet", "O arquivo config.yaml não foi atualizado corretamente.")    
        
if __name__ == "__main__":
    unittest.main()
