import pandas as pd
import yaml
import logging
import time
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, RandomizedSearchCV, StratifiedKFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score, f1_score
from imblearn.over_sampling import SMOTE
import joblib
import numpy as np
from scipy.stats import randint

# Registrar a hora inicial do processamento
start_time = time.time()

# Configurar logging
with open(r'D:\Github\data-science\projetos\creditcard_fraud_detection\creditcard\config\config.yaml', 'r') as file:
    config = yaml.safe_load(file)

logging.basicConfig(filename=config['paths']['logs_path'] + 'modeling_random_forest.log', level=logging.INFO, format='%(asctime)s - %(message)s')

# Carregar os dados processados e normalizados
df_normalized = pd.read_csv(config['data']['processed_data_path'])

# Verificar e tratar valores NaN na variável de destino 'Class'
if df_normalized['Class'].isnull().sum() > 0:
    logging.info(f"Valores NaN encontrados em 'Class': {df_normalized['Class'].isnull().sum()}")
    df_normalized['Class'].fillna(df_normalized['Class'].mode()[0], inplace=True)

# Dividir o dataset em conjunto de treinamento e teste (70/30)
X = df_normalized.drop(columns=['Class'])
y = df_normalized['Class']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=config['model']['params']['test_size'], random_state=config['model']['params']['random_state'])

# Verificar a distribuição das classes no conjunto de treinamento
logging.info("Distribuição das classes no conjunto de treinamento antes do SMOTE:")
logging.info(y_train.value_counts())

# Lidar com dados desbalanceados utilizando SMOTE
sm = SMOTE(random_state=config['model']['params']['random_state'])
X_train_res, y_train_res = sm.fit_resample(X_train, y_train)

# Verificar a distribuição das classes após o SMOTE
logging.info("Distribuição das classes no conjunto de treinamento após o SMOTE:")
logging.info(y_train_res.value_counts())

# Treinar o modelo de Random Forest
rf_model = RandomForestClassifier(random_state=config['model']['params']['random_state'])
rf_model.fit(X_train_res, y_train_res)
y_pred_rf = rf_model.predict(X_test)
logging.info("Relatório de Classificação do modelo de Random Forest:")
logging.info(classification_report(y_test, y_pred_rf, zero_division=0))
logging.info(f"AUC-ROC: {roc_auc_score(y_test, y_pred_rf)}")

# Refine a grade de hiperparâmetros
param_dist = {
    'n_estimators': randint(50, 500),
    'max_depth': [None, 10, 20, 30, 40, 50],
    'min_samples_split': randint(2, 20),
    'min_samples_leaf': randint(1, 10),
    'max_features': ['sqrt', 'log2'],  # Removido 'auto'
    'bootstrap': [True, False],
    'class_weight': [{0: 1, 1: w} for w in [10, 20, 30, 40, 50]]
}

# Utilizar RandomizedSearchCV para uma busca mais eficiente
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=config['model']['params']['random_state'])
random_search = RandomizedSearchCV(rf_model, param_distributions=param_dist, n_iter=100, cv=cv, scoring='roc_auc', random_state=config['model']['params']['random_state'])
random_search.fit(X_train_res, y_train_res)
best_rf_model = random_search.best_estimator_
logging.info(f"Melhores hiperparâmetros em treino: {random_search.best_params_}")

# Aplicar os melhores hiperparâmetros em dados de treino
y_train_pred_rf = best_rf_model.predict(X_train_res)
logging.info("Relatório de Classificação nos dados de treino com melhores hiperparâmetros:")
logging.info(classification_report(y_train_res, y_train_pred_rf, zero_division=0))
logging.info(f"AUC-ROC: {roc_auc_score(y_train_res, y_train_pred_rf)}")

# Aplicar os melhores hiperparâmetros nos dados de teste
y_test_pred_rf = best_rf_model.predict(X_test)
logging.info("Relatório de Classificação nos dados de teste com melhores hiperparâmetros:")
logging.info(classification_report(y_test, y_test_pred_rf, zero_division=0))
logging.info(f"AUC-ROC: {roc_auc_score(y_test, y_test_pred_rf)}")

# Realizar validação cruzada estratificada nos dados de treino
cross_val_scores_train = cross_val_score(best_rf_model, X_train_res, y_train_res, cv=cv, scoring='roc_auc')
logging.info("Validação cruzada estratificada nos dados de treino:")
logging.info(cross_val_scores_train)
logging.info(f"AUC-ROC médio: {cross_val_scores_train.mean()}")

# Realizar validação cruzada estratificada nos dados de teste
cross_val_scores_test = cross_val_score(best_rf_model, X_test, y_test, cv=cv, scoring='roc_auc')
logging.info("Validação cruzada estratificada nos dados de teste:")
logging.info(cross_val_scores_test)
logging.info(f"AUC-ROC médio: {cross_val_scores_test.mean()}")

# Encontrar o melhor threshold para o conjunto de treino
y_train_proba = best_rf_model.predict_proba(X_train_res)[:, 1]
thresholds = np.arange(0.0, 1.0, 0.01)
f1_scores = []

for threshold in thresholds:
    y_train_pred_threshold = (y_train_proba >= threshold).astype(int)
    f1 = f1_score(y_train_res, y_train_pred_threshold)
    f1_scores.append(f1)

best_threshold = thresholds[np.argmax(f1_scores)]
best_f1_score = max(f1_scores)

logging.info(f"Melhor threshold: {best_threshold}")
logging.info(f"Melhor F1-score no treino: {best_f1_score}")

# Aplicar o melhor threshold para o conjunto de teste
y_test_proba = best_rf_model.predict_proba(X_test)[:, 1]
y_test_pred_best_threshold = (y_test_proba >= best_threshold).astype(int)
logging.info("Relatório de Classificação no conjunto de teste com o melhor threshold:")
logging.info(classification_report(y_test, y_test_pred_best_threshold, zero_division=0))
logging.info(f"AUC-ROC: {roc_auc_score(y_test, y_test_pred_best_threshold)}")

# Salvar o melhor modelo após a aplicação do threshold no conjunto de teste
best_model_with_threshold = {'model': best_rf_model, 'threshold': best_threshold}
joblib.dump(best_model_with_threshold, config['paths']['models_path'] + 'best_rf_model_with_threshold.pkl')
joblib.dump(sm, config['paths']['preprocessors_path'] + 'smote.pkl')

logging.info("Melhor modelo salvo como 'best_rf_model_with_threshold.pkl'.")
logging.info("SMOTE salvo como 'smote.pkl'.")

# Registrar a hora final do processamento
end_time = time.time()
total_time = end_time - start_time
hours, remainder = divmod(total_time, 3600)
minutes, seconds = divmod(remainder, 60)
logging.info(f"Tempo total de processamento: {int(hours)} horas, {int(minutes)} minutos e {int(seconds)} segundos.")

print("4 - Etapa de Modelagem concluída.")

# --- Análise Segmentada ---

# Adicionar as previsões e probabilidades ao DataFrame
df_normalized['predicted_class'] = best_rf_model.predict(X)
df_normalized['predicted_proba'] = best_rf_model.predict_proba(X)[:, 1]

# Analisar desempenho por segmentos (ex: valor da transação)
df_normalized['transaction_amount_category'] = pd.cut(df_normalized['Amount'], bins=[0, 50, 100, 200, 500, 1000, 5000, 10000], labels=['0-50', '50-100', '100-200', '200-500', '500-1000', '1000-5000', '5000-10000'])

# Filtrar categorias válidas antes de calcular as métricas
valid_categories = df_normalized[df_normalized['transaction_amount_category'].notna()]

# Verificar as categorias presentes
categorias_existentes = df_normalized['transaction_amount_category'].value_counts()
print("Categorias de valor de transação presentes nos dados:")
print(categorias_existentes)

# Criar um gráfico de barras para visualizar o desempenho por categoria de valor da transação
plt.figure(figsize=(10, 6))
sns.barplot(x='transaction_amount_category', y='Class', data=valid_categories, estimator=lambda x: sum(x == 1) / len(x))
plt.title('Proporção de Fraudes por Categoria de Valor da Transação')
plt.xlabel('Categoria de Valor da Transação')
plt.ylabel('Proporção de Fraudes')
plt.savefig(r'D:\Github\data-science\projetos\creditcard_fraud_detection\creditcard\reports\figures\proporcao_fraudes_categoria_valor_transacao.png')
plt.show()

# Função para ajustar o threshold para cada categoria
def ajustar_threshold_para_categoria(categoria, df):
    subset = df[df['transaction_amount_category'] == categoria]
    y_true = subset['Class']
    y_proba = subset['predicted_proba']
    
    # Verificar se há amostras suficientes
    if len(y_true) == 0 or y_true.nunique() < 2:
        print(f"Categoria {categoria}: amostras insuficientes para ajustar threshold.")
        return
    
    thresholds = np.arange(0.0, 1.0, 0.01)
    f1_scores = []

    for threshold in thresholds:
        y_pred_threshold = (y_proba >= threshold).astype(int)
        f1 = f1_score(y_true, y_pred_threshold, zero_division=1)
        f1_scores.append(f1)

    best_threshold = thresholds[np.argmax(f1_scores)]
    best_f1_score = max(f1_scores)

    print(f"Melhor threshold para a categoria {categoria}: {best_threshold}")
    print(f"Melhor F1-score para a categoria {categoria}: {best_f1_score}")

    # Aplicar o melhor threshold para a categoria
    y_pred_best_threshold = (y_proba >= best_threshold).astype(int)
    print(classification_report(y_true, y_pred_best_threshold, zero_division=1))
    print(f"AUC-ROC para a categoria {categoria}: {roc_auc_score(y_true, y_proba)}")

# Aplicar a função de ajuste de threshold para todas as categorias
categorias = valid_categories['transaction_amount_category'].unique()
for categoria in categorias:
    ajustar_threshold_para_categoria(categoria, valid_categories)

print("Etapa 6: Análise Segmentada concluída.")
