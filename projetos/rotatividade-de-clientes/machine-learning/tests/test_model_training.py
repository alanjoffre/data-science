import unittest
from unittest.mock import patch, MagicMock
import pandas as pd
import yaml
from model_training import (
    carregar_configuracao,
    validar_colunas,
    carregar_dados,
    tratar_dados,
    criar_preprocessador,
    aplicar_smote,
    salvar_preprocessador
)

class TestModelTraining(unittest.TestCase):

    @patch("model_training.yaml.safe_load")
    @patch("builtins.open")
    def test_carregar_configuracao(self, mock_open, mock_safe_load):
        # Configuração simulada
        mock_safe_load.return_value = {'data': {'raw': 'test_path.csv'}}
        
        # Teste de carregamento de configuração
        config = carregar_configuracao()
        self.assertEqual(config['data']['raw'], 'test_path.csv')
    
    def test_validar_colunas(self):
        # DataFrame de teste
        df = pd.DataFrame({
            'Balance': [1000, 2000],
            'Age': [30, 40],
            'CreditScore': [600, 700],
            'Tenure': [3, 4],
            'EstimatedSalary': [50000, 60000],
            'Exited': [0, 1]
        })
        # Colunas esperadas
        required_cols = ['Balance', 'Age', 'CreditScore', 'Tenure', 'EstimatedSalary', 'Exited']
        
        # Teste de validação das colunas
        try:
            validar_colunas(df, required_cols)
        except ValueError:
            self.fail("validar_colunas() falhou ao validar colunas presentes no DataFrame.")

    @patch("pandas.read_csv")
    def test_carregar_dados(self, mock_read_csv):
        # Simular carregamento do CSV
        mock_df = pd.DataFrame({'Balance': [1000], 'Exited': [1]})
        mock_read_csv.return_value = mock_df
        config = {'data': {'raw': 'test_path.csv'}}
        
        df = carregar_dados(config)
        pd.testing.assert_frame_equal(df, mock_df)

    def test_tratar_dados(self):
        # Configuração e dados de entrada simulados
        df = pd.DataFrame({
            'Balance': [1000, None, 2000],
            'Age': [30, 40, 25],
            'CreditScore': [600, None, 700],
            'Tenure': [3, 4, 3],
            'EstimatedSalary': [50000, 60000, None],
            'Exited': [0, 1, 0]
        })
        num_cols = ['Balance', 'CreditScore', 'EstimatedSalary']
        config = {'data': {'data_processed': 'dummy_path.csv'}}
        
        # Teste de tratamento de dados
        df_processed = tratar_dados(df, num_cols, config)
        self.assertFalse(df_processed.isnull().any().any(), "Dados ainda contêm valores ausentes após o tratamento.")

    def test_criar_preprocessador(self):
        num_cols = ['Balance', 'CreditScore', 'EstimatedSalary']
        preprocessor = criar_preprocessador(num_cols)
        self.assertIsNotNone(preprocessor, "Preprocessador não foi criado corretamente.")

    def test_aplicar_smote(self):
        # Dados de teste
        X = pd.DataFrame({'Feature': [1, 2, 3, 4], 'Balance': [100, 200, 300, 400]})
        y = pd.Series([0, 0, 1, 1])
        
        # Aplicar SMOTE
        X_resampled, y_resampled = aplicar_smote(X, y)
        
        # Verificação do balanceamento
        self.assertEqual(len(X_resampled), len(y_resampled))
        self.assertTrue((y_resampled.value_counts() == 2).all(), "Classes não foram balanceadas corretamente.")

    @patch("joblib.dump")
    def test_salvar_preprocessador(self, mock_dump):
        preprocessor = MagicMock()
        salvar_preprocessador(preprocessor, "dummy_path.joblib")
        
        # Verificar se o joblib.dump foi chamado corretamente
        mock_dump.assert_called_once_with(preprocessor, "dummy_path.joblib")

if __name__ == "__main__":
    unittest.main()
