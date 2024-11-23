import unittest
import os
import yaml

class TestCarregamentoDados(unittest.TestCase):
    def setUp(self):
        # Atualizar o arquivo de configuração
        config_path = 'D:\\Github\\data-science\\projetos\\analise-de-sentimentos\\nlp\\config\\config.yaml'
        with open(config_path, 'r') as file:
            self.config = yaml.safe_load(file)

        # Garantir que os campos necessários estão presentes
        self.config['directories']['raw_data'] = 'D:\\Github\\data-science\\projetos\\analise-de-sentimentos\\nlp\\data\\raw\\'
        self.config['files']['raw_dataset'] = 'asentimentos.parquet'
        self.config['files']['log_file'] = 'etapa1_carregamento_dados.log'
        with open(config_path, 'w') as file:
            yaml.safe_dump(self.config, file)

    def test_carregar_dados(self):
        raw_data_path = os.path.join(self.config['directories']['raw_data'], self.config['files']['raw_dataset'])
        self.assertTrue(os.path.exists(raw_data_path), "O arquivo de dados brutos não existe.")

if __name__ == '__main__':
    unittest.main()
