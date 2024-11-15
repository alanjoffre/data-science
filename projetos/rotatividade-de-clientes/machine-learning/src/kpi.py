import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import xgboost as xgb
import joblib
import numpy as np

# Passo 1: Carregar o arquivo pré-processado
file_path = 'D:\\Github\\data-science\\projetos\\rotatividade-de-clientes\\machine-learning\\data\\processed\\rclientes_preprocessado.csv'
clientes_df = pd.read_csv(file_path)

# Passo 2: Separar as variáveis independentes (X) e a variável alvo (y)
X = clientes_df.drop(columns=['Exited'])  # Supondo que 'Exited' seja a coluna de saída
y = clientes_df['Exited']

# Passo 3: Dividir os dados em conjunto de treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Passo 4: Carregar o modelo XGBoost treinado
modelo_path = 'D:\\Github\\data-science\\projetos\\rotatividade-de-clientes\\machine-learning\\models\\final_model.joblib'
modelo = joblib.load(modelo_path)

# Passo 5: Realizar previsões
y_pred = modelo.predict(X_test)
y_pred_proba = modelo.predict_proba(X_test)[:, 1]

# Passo 6: Calcular os KPIs

# Acurácia, Precisão, Revocação, F1-Score, AUC-ROC
acuracia = accuracy_score(y_test, y_pred)
precisao = precision_score(y_test, y_pred)
revocacao = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
auc_roc = roc_auc_score(y_test, y_pred_proba)
matriz_confusao = confusion_matrix(y_test, y_pred)

# 1. Taxa de Falsos Positivos (FPR)
TN, FP, FN, TP = matriz_confusao.ravel()
FPR = FP / (FP + TN)

# 2. Taxa de Falsos Negativos (FNR)
FNR = FN / (FN + TP)

# 3. Acurácia Balanceada
TPR = TP / (TP + FN)  # Revocação (Sensibilidade)
TNR = TN / (TN + FP)  # Especificidade
balanced_accuracy = (TPR + TNR) / 2

# 4. Matriz de Confusão Normalizada
cm_normalized = matriz_confusao.astype('float') / matriz_confusao.sum(axis=1)[:, np.newaxis]

# 5. Kappa de Cohen
accuracy_observed = (TP + TN) / (TP + TN + FP + FN)
accuracy_expected = ((TP + FP) / (TP + TN + FP + FN)) * ((TP + FN) / (TP + TN + FP + FN)) + ((TN + FN) / (TP + TN + FP + FN)) * ((TN + FP) / (TP + TN + FP + FN))
kappa = (accuracy_observed - accuracy_expected) / (1 - accuracy_expected)

# 6. Gini Coefficient
gini = 2 * auc_roc - 1

# 7. Lift (usando probabilidades de previsão)
def lift_function(y_true, y_pred_probs):
    data = pd.DataFrame({'y_true': y_true, 'y_pred_probs': y_pred_probs})
    data = data.sort_values(by='y_pred_probs', ascending=False)
    data['cumulative_true'] = data['y_true'].cumsum()
    data['cumulative_total'] = np.arange(1, len(data) + 1)
    return data['cumulative_true'] / data['cumulative_total']

lift = lift_function(y_test, y_pred_proba).mean()

# Passo 7: Exibir os resultados
print(f'Acurácia: {acuracia:.2f}')
print(f'Precisão: {precisao:.2f}')
print(f'Revocação: {revocacao:.2f}')
print(f'F1-Score: {f1:.2f}')
print(f'AUC-ROC: {auc_roc:.2f}')
print(f'Matriz de Confusão:\n{matriz_confusao}')

print(f'\nKPIs Adicionais:')
print(f'Taxa de Falsos Positivos (FPR): {FPR:.2f}')
print(f'Taxa de Falsos Negativos (FNR): {FNR:.2f}')
print(f'Acurácia Balanceada: {balanced_accuracy:.2f}')
print(f'Matriz de Confusão Normalizada:\n{cm_normalized}')
print(f'Kappa de Cohen: {kappa:.2f}')
print(f'Índice de Gini: {gini:.2f}')
print(f'Lift: {lift:.2f}')