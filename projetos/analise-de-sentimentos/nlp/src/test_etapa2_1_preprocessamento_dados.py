import unittest
import os
import yaml
import dask.dataframe as dd

class TestPreprocessamentoDados(unittest.TestCase):
    def setUp(self):
        # Carregar o arquivo de configuração
        with open('D:\\Github\\data-science\\projetos\\analise-de-sentimentos\\nlp\\config\\config.yaml', 'r') as file:
            self.config = yaml.safe_load(file)
        
        # Atualizar campos específicos para o teste
        self.config['files']['log_file'] = "etapa2_1_preprocessamento_dados.log"
        self.config['files']['processed_dataset'] = "etapa2_1_preprocessamento_dados.parquet"
        self.config['files']['processed_dataset_csv'] = "etapa2_1_preprocessamento_dados.csv"
        self.config['files']['raw_dataset'] = "asentimentos.parquet"
        with open('D:\\Github\\data-science\\projetos\\analise-de-sentimentos\\nlp\\config\\config.yaml', 'w') as file:
            yaml.safe_dump(self.config, file)

    def test_preprocessar_dados(self):
        processed_data_path = os.path.join(self.config['directories']['processed_data'], self.config['files']['processed_dataset'])
        preprocessed_data_path = os.path.join(self.config['directories']['processed_data'], "etapa2_1_preprocessamento_dados.parquet")

        # Testar se o dataset processado foi carregado
        df = dd.read_parquet(processed_data_path)
        self.assertGreater(len(df), 0, "O dataset processado não foi carregado corretamente.")

        # Testar se o dataset pré-processado foi salvo
        df_preprocessed = dd.read_parquet(preprocessed_data_path)
        self.assertGreater(len(df_preprocessed), 0, "O dataset pré-processado não foi salvo corretamente.")    

        # Verificar se os textos foram limpos corretamente
        tweets = df_preprocessed['tweet'].head()
        for tweet in tweets:
            self.assertNotRegex(tweet, r"http\S+", "Não removeu os links corretamente.")
            self.assertNotRegex(tweet, r"@\w+", "Não removeu as menções corretamente.")
            self.assertNotRegex(tweet, r"#\w+", "Não removeu as hashtags corretamente.")
            self.assertNotRegex(tweet, r"\d+", "Não removeu os números corretamente.")
            self.assertNotRegex(tweet, r"[^\w\s]", "Não removeu os caracteres especiais corretamente.")

        # Testar se o config.yaml foi atualizado
        with open('D:\\Github\\data-science\\projetos\\analise-de-sentimentos\\nlp\\config\\config.yaml', 'r') as file:
            config_updated = yaml.safe_load(file)
        
        self.assertEqual(config_updated['files']['log_file'], "etapa2_1_preprocessamento_dados.log", "O arquivo config.yaml não foi atualizado corretamente.")
        self.assertEqual(config_updated['files']['processed_dataset'], "etapa2_1_preprocessamento_dados.parquet", "O arquivo config.yaml não foi atualizado corretamente.")
        self.assertEqual(config_updated['files']['processed_dataset_csv'], "etapa2_1_preprocessamento_dados.csv", "O arquivo config.yaml não foi atualizado corretamente.")
        self.assertEqual(config_updated['files']['raw_dataset'], "asentimentos.parquet", "O arquivo config.yaml não foi atualizado corretamente.")
        
if __name__ == '__main__':
    unittest.main()
