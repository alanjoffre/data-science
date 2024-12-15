import unittest
import pandas as pd
import yaml
import os
from data_preprocessing import carregar_configuracao, validar_colunas, carregar_dados, tratar_dados, criar_preprocessador, aplicar_smote

class TestPreprocessamentoDados(unittest.TestCase):

    def setUp(self):
        # Configuração inicial para os testes
        self.config_path = 'D:\\Github\\data-science\\projetos\\rotatividade-de-clientes\\machine-learning\\config\\config.yaml'
        # Adicionando mais amostras para evitar erro com SMOTE
        self.sample_data = pd.DataFrame({
            'Balance': [1000, 2000, 1500, 2500, 3000, 3500],
            'Age': [30, 40, 35, 45, 50, 55],
            'CreditScore': [600, 700, 650, 720, 680, 690],
            'Tenure': [5, 10, 7, 8, 6, 9],
            'EstimatedSalary': [50000, 60000, 55000, 62000, 58000, 57000],
            'Exited': [0, 1, 0, 1, 0, 1]
        })

    def test_carregar_configuracao(self):
        config = carregar_configuracao(self.config_path)
        self.assertIn('data', config, "Configuração deve conter a chave 'data'")

    def test_validar_colunas(self):
        required_cols = ['Balance', 'Age', 'CreditScore', 'Tenure', 'EstimatedSalary', 'Exited']
        # Teste não deve levantar exceções
        validar_colunas(self.sample_data, required_cols)

    def test_carregar_dados(self):
        config = carregar_configuracao(self.config_path)
        df = carregar_dados(config)
        self.assertIsInstance(df, pd.DataFrame, "Os dados carregados devem ser um DataFrame")

    def test_tratar_dados(self):
        config = carregar_configuracao(self.config_path)
        num_cols = ['Balance', 'Age', 'CreditScore', 'Tenure', 'EstimatedSalary']
        treated_df = tratar_dados(self.sample_data.copy(), num_cols, config)
        self.assertFalse(treated_df.isnull().values.any(), "DataFrame tratado não deve conter valores nulos")

    def test_criar_preprocessador(self):
        num_cols = ['Balance', 'Age', 'CreditScore', 'Tenure', 'EstimatedSalary']
        preprocessor = criar_preprocessador(num_cols)
        self.assertIsNotNone(preprocessor, "Pré-processador deve ser criado com sucesso")

    def test_aplicar_smote(self):
        X = self.sample_data.drop('Exited', axis=1)
        y = self.sample_data['Exited']
        # Ajuste do k_neighbors para o número mínimo de amostras da classe minoritária
        X_resampled, y_resampled = aplicar_smote(X, y)
        self.assertEqual(X_resampled.shape[0], y_resampled.shape[0], "X e y devem ter o mesmo número de linhas após SMOTE")

if __name__ == '__main__':
    unittest.main()
