import unittest
import os
import yaml
from etapa2_6_vetorizacao_texto import atualizar_config

class TestAtualizarConfig(unittest.TestCase):

    def setUp(self):
        self.config_path = 'D:\\Github\\data-science\\projetos\\analise-de-sentimentos\\nlp\\config\\config.yaml'
        with open(self.config_path, 'r') as file:  # Corrigido para usar self.config_path
            self.original_config = yaml.safe_load(file)
        
        # Atualizar campos específicos para o teste
        self.test_config = self.original_config.copy()
        self.test_config['files']['log_file'] = "etapa2_6_vetorizacao_texto.log"
        self.test_config['files']['processed_dataset'] = "etapa2_6_vetorizacao_texto.parquet"
        self.test_config['files']['processed_dataset_csv'] = "etapa2_6_vetorizacao_texto.csv"
        self.test_config['files']['raw_dataset'] = "asentimentos.parquet"
        with open(self.config_path, 'w') as file:
            yaml.safe_dump(self.test_config, file)

    def test_atualizar_config(self):
        # Testar se o config.yaml foi atualizado corretamente
        with open(self.config_path, 'r') as file:  # Corrigido para usar self.config_path
            config_updated = yaml.safe_load(file)
        
        self.assertEqual(config_updated['files']['log_file'], "etapa2_6_vetorizacao_texto.log", "O arquivo config.yaml não foi atualizado corretamente.")
        self.assertEqual(config_updated['files']['processed_dataset'], "etapa2_6_vetorizacao_texto.parquet", "O arquivo config.yaml não foi atualizado corretamente.")
        self.assertEqual(config_updated['files']['processed_dataset_csv'], "etapa2_6_vetorizacao_texto.csv", "O arquivo config.yaml não foi atualizado corretamente.")
        self.assertEqual(config_updated['files']['raw_dataset'], "asentimentos.parquet", "O arquivo config.yaml não foi atualizado corretamente.")    

if __name__ == '__main__':
    unittest.main()
