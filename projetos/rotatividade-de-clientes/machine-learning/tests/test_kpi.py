import unittest
from unittest.mock import patch, MagicMock
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix
import joblib
import kpi  # Importa o módulo que contém o código KPI

class TestKPI(unittest.TestCase):
    @patch("kpi.pd.read_csv")
    @patch("kpi.joblib.load")
    def setUp(self, mock_load_model, mock_read_csv):
        # Mock do dataset
        self.mock_data = {
            'Feature1': [0.1, 0.2, 0.3, 0.4, 0.5],
            'Feature2': [1, 0, 1, 0, 1],
            'Exited': [0, 1, 0, 1, 0]
        }
        self.clientes_df = pd.DataFrame(self.mock_data)
        
        # Mock do método read_csv
        mock_read_csv.return_value = self.clientes_df
        
        # Mock do modelo
        mock_model = MagicMock()
        mock_model.predict.return_value = np.array([0, 1, 0, 1, 0])
        mock_model.predict_proba.return_value = np.array([[0.7, 0.3], [0.2, 0.8], [0.6, 0.4], [0.1, 0.9], [0.8, 0.2]])
        mock_load_model.return_value = mock_model
        
        # Carregar o modelo e dados mockados
        self.model = joblib.load('dummy_path')
        self.X_test = self.clientes_df.drop(columns=['Exited'])
        self.y_test = self.clientes_df['Exited']

    def test_kpis(self):
        # Executa as previsões
        y_pred = self.model.predict(self.X_test)
        y_pred_proba = self.model.predict_proba(self.X_test)[:, 1]
        
        # Calcula as métricas
        acuracia = kpi.accuracy_score(self.y_test, y_pred)
        precisao = kpi.precision_score(self.y_test, y_pred)
        revocacao = kpi.recall_score(self.y_test, y_pred)
        f1 = kpi.f1_score(self.y_test, y_pred)
        auc_roc = kpi.roc_auc_score(self.y_test, y_pred_proba)
        
        matriz_confusao = confusion_matrix(self.y_test, y_pred)
        TN, FP, FN, TP = matriz_confusao.ravel()
        
        # KPIs adicionais
        FPR = FP / (FP + TN)
        FNR = FN / (FN + TP)
        TPR = TP / (TP + FN)
        TNR = TN / (TN + FP)
        balanced_accuracy = (TPR + TNR) / 2
        cm_normalized = matriz_confusao.astype('float') / matriz_confusao.sum(axis=1)[:, np.newaxis]
        
        accuracy_observed = (TP + TN) / (TP + TN + FP + FN)
        accuracy_expected = ((TP + FP) / (TP + TN + FP + FN)) * ((TP + FN) / (TP + TN + FP + FN)) + ((TN + FN) / (TP + TN + FP + FN)) * ((TN + FP) / (TP + TN + FP + FN))
        kappa = (accuracy_observed - accuracy_expected) / (1 - accuracy_expected)
        
        gini = 2 * auc_roc - 1
        
        lift = kpi.lift_function(self.y_test, y_pred_proba).mean()
        
        # Verifica se as métricas estão dentro dos limites esperados
        self.assertGreaterEqual(acuracia, 0)
        self.assertLessEqual(acuracia, 1)
        self.assertGreaterEqual(precisao, 0)
        self.assertLessEqual(precisao, 1)
        self.assertGreaterEqual(revocacao, 0)
        self.assertLessEqual(revocacao, 1)
        self.assertGreaterEqual(f1, 0)
        self.assertLessEqual(f1, 1)
        self.assertGreaterEqual(auc_roc, 0)
        self.assertLessEqual(auc_roc, 1)
        self.assertGreaterEqual(kappa, -1)
        self.assertLessEqual(kappa, 1)
        self.assertGreaterEqual(gini, -1)
        self.assertLessEqual(gini, 1)
        self.assertGreater(lift, 0)

if __name__ == '__main__':
    unittest.main()
