import unittest
from unittest.mock import patch, MagicMock
import pandas as pd
from evaluation import carregar_configuracao, carregar_dados_e_modelo, avaliar_modelo, main
from sklearn.metrics import classification_report

class TestEvaluation(unittest.TestCase):

    @patch("builtins.open", new_callable=MagicMock)
    @patch("yaml.safe_load", return_value={'data': {'processed': 'fake_path.csv'}, 'models': {'final_model': 'fake_model.joblib'}, 'reports': {'classification_final': 'report.txt', 'classification_threshold': 'report_optimal.txt'}})
    def test_carregar_configuracao(self, mock_yaml, mock_open):
        mock_open.return_value.__enter__.return_value = MagicMock()
        config = carregar_configuracao("fake_path.yaml")
        
        # Testa se o mock carregou as configurações corretamente
        self.assertIsInstance(config, dict)
        self.assertIn('data', config)
        self.assertEqual(config['data']['processed'], 'fake_path.csv')
        mock_yaml.assert_called_once()

    @patch("builtins.open", new_callable=MagicMock)
    @patch("joblib.load", return_value=MagicMock())
    @patch("pandas.read_csv", return_value=pd.DataFrame({'Exited': [0, 1, 0, 1], 'Age': [25, 30, 35, 40]}))
    def test_carregar_dados_e_modelo(self, mock_read_csv, mock_joblib_load, mock_open):
        # Mockando as funções de open e leitura de arquivos
        mock_open.return_value.__enter__.return_value = MagicMock()
        
        config = {'data': {'processed': 'fake_path.csv'}, 'models': {'final_model': 'fake_model.joblib'}}
        
        df, model = carregar_dados_e_modelo(config)
        
        self.assertIsInstance(df, pd.DataFrame)
        self.assertTrue(model)
        mock_read_csv.assert_called_once_with('fake_path.csv')
        mock_joblib_load.assert_called_once_with('fake_model.joblib')

    @patch("builtins.open", new_callable=MagicMock)
    @patch("joblib.load", return_value=MagicMock())
    @patch("pandas.read_csv", return_value=pd.DataFrame({'Exited': [0, 1, 0, 1], 'Age': [25, 30, 35, 40]}))
    @patch("sklearn.metrics.classification_report", return_value="classification_report_mock")
    @patch("sklearn.metrics.roc_curve", return_value=([0, 0.1, 1], [0, 0.9, 1], [0.1, 0.2, 0.3]))
    @patch("sklearn.metrics.auc", return_value=0.85)
    def test_avaliar_modelo(self, mock_auc, mock_roc_curve, mock_classification_report, mock_read_csv, mock_joblib_load, mock_open):
        config = {'reports': {'classification_final': 'report.txt', 'classification_threshold': 'report_optimal.txt'}}
        
        # Ajustando o mock do modelo para retornar valores esperados
        model = MagicMock()
        model.predict.return_value = [0, 1, 0, 1]
        model.predict_proba.return_value = [[0.7, 0.3], [0.2, 0.8], [0.6, 0.4], [0.1, 0.9]]  # Retornando um array válido para probabilidade
        
        df = pd.DataFrame({'Exited': [0, 1, 0, 1], 'Age': [25, 30, 35, 40]})
        X_test = df.drop('Exited', axis=1)
        y_test = df['Exited']
        
        # Chamando a função que estamos testando
        avaliar_modelo(model, X_test, y_test, config)

        # Verificando se as funções de escrita de arquivo foram chamadas
        mock_open.assert_any_call('report.txt', 'w')
        mock_open.assert_any_call('report_optimal.txt', 'w')
        
        # Verificando se os métodos de métrica foram chamados corretamente
        mock_classification_report.assert_called()
        mock_roc_curve.assert_called()
        mock_auc.assert_called()

    @patch("builtins.open", new_callable=MagicMock)
    @patch("yaml.safe_load", return_value={'data': {'processed': 'fake_path.csv'}, 'models': {'final_model': 'fake_model.joblib'}, 'reports': {'classification_final': 'report.txt', 'classification_threshold': 'report_optimal.txt'}})
    @patch("pandas.read_csv", return_value=pd.DataFrame({'Exited': [0, 1, 0, 1], 'Age': [25, 30, 35, 40]}))
    @patch("joblib.load", return_value=MagicMock())
    def test_main(self, mock_joblib_load, mock_read_csv, mock_yaml, mock_open):
        with self.assertLogs('evaluation', level='INFO') as log:
            main()
            self.assertIn('Configuração carregada com sucesso.', log.output)
            self.assertIn('Dados e modelo carregados com sucesso.', log.output)

if __name__ == '__main__':
    unittest.main()
