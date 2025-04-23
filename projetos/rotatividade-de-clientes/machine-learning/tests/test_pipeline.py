import unittest
import os
import joblib
import pandas as pd
from pipeline import carregar_configuracao, preprocessamento_dados, treinar_modelo, avaliar_modelo, salvar_modelo

class TestPipeline(unittest.TestCase):
    
    def setUp(self):
        # Configurações antes de cada teste
        self.config_path = 'D:\\Github\\data-science\\projetos\\rotatividade-de-clientes\\machine-learning\\config\\config.yaml'
        self.model_save_path = 'D:\\Github\\data-science\\projetos\\rotatividade-de-clientes\\machine-learning\\model_test.joblib'
        
        # Simulando dados de entrada para testes
        self.X_train = pd.DataFrame({
            'feature1': [0.5, 0.3, 0.1],
            'feature2': [1.5, 1.2, 1.1]
        })
        self.y_train = pd.Series([0, 1, 0])
        self.X_test = pd.DataFrame({
            'feature1': [0.4, 0.2],
            'feature2': [1.4, 1.0]
        })
        self.y_test = pd.Series([1, 0])
        
    def test_carregar_configuracao(self):
        # Teste para carregar a configuração
        config = carregar_configuracao(self.config_path)
        self.assertIn('models', config)  # Verifica se a chave 'models' está presente na configuração

    def test_preprocessamento_dados(self):
        # Teste de pré-processamento de dados
        X_train, X_test, y_train, y_test = preprocessamento_dados()
        self.assertIsInstance(X_train, pd.DataFrame)
        self.assertIsInstance(X_test, pd.DataFrame)
        self.assertIsInstance(y_train, pd.Series)
        self.assertIsInstance(y_test, pd.Series)
        
    def test_treinar_modelo(self):
        # Teste para treinamento do modelo
        modelo = treinar_modelo(self.X_train, self.y_train)
        self.assertIsNotNone(modelo)  # Verifica se o modelo foi treinado com sucesso
    
    def test_avaliar_modelo(self):
        # Teste para avaliação do modelo
        modelo = treinar_modelo(self.X_train, self.y_train)
        try:
            avaliar_modelo(modelo, self.X_test, self.y_test)
            success = True
        except Exception as e:
            success = False
        self.assertTrue(success)  # Verifica se a avaliação foi bem-sucedida

    def test_salvar_modelo(self):
        # Teste para salvar o modelo
        modelo = treinar_modelo(self.X_train, self.y_train)
        salvar_modelo(modelo, self.model_save_path)
        self.assertTrue(os.path.exists(self.model_save_path))  # Verifica se o arquivo foi salvo com sucesso

    def tearDown(self):
        # Remover arquivos de teste após cada execução de teste
        if os.path.exists(self.model_save_path):
            os.remove(self.model_save_path)

if __name__ == "__main__":
    unittest.main()
