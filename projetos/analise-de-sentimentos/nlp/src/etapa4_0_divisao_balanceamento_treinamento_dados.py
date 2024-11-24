import dask.dataframe as dd
import pandas as pd
import yaml
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier, ExtraTreesClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix
import xgboost as xgb
import lightgbm as lgb
import numpy as np
import os

# Carregar arquivo de configuração
config_path = 'D:\\Github\\data-science\\projetos\\analise-de-sentimentos\\nlp\\config\\config.yaml'
with open(config_path, 'r') as file:
    config = yaml.safe_load(file)

# Carregar o dataset
dataset_path = config['directories']['processed_data'] + config['files']['processed_dataset']
df = dd.read_parquet(dataset_path)

# Convertendo para pandas para usar train_test_split
df_pandas = df.compute()

# Verificar a contagem de linhas antes do balanceamento
print("Contagem de linhas por sentimento_codificado antes do balanceamento:")
print(df_pandas['sentimento_codificado'].value_counts())

# Selecionar colunas de features relevantes, assumindo que 'vetorizacao_tweet' existe
if 'vetorizacao_tweet' in df_pandas.columns:
    df_pandas['vetorizacao_tweet'] = df_pandas['vetorizacao_tweet'].apply(eval)
    vetorizacao_tweet = np.stack(df_pandas['vetorizacao_tweet'].values)
    
    # Selecionar features vetorizadas e o target
    X = vetorizacao_tweet
    y = df_pandas['sentimento_codificado']
else:
    raise ValueError("A coluna 'vetorizacao_tweet' não está presente no dataset.")

# Remover possíveis valores NaN do target
y = y.dropna()

# Remover linhas em X que correspondem a valores NaN no target
X = X[y.index]

# Verificar a integridade dos dados antes do balanceamento
print("\nVerificação de integridade dos dados antes do balanceamento:")
print(f"Shape de X: {X.shape}")
print(f"Shape de y: {y.shape}")

# Definir o pipeline para oversampling e undersampling
over = SMOTE(sampling_strategy=0.5)  # Aumentar a minoria para 50% da classe majoritária
under = RandomUnderSampler(sampling_strategy=0.5)  # Reduzir a maioria para 50% da classe minoritária
steps = [('over', over), ('under', under)]
pipeline = Pipeline(steps=steps)

# Aplicar o pipeline de balanceamento
X_resampled, y_resampled = pipeline.fit_resample(X, y)

# Reconstruir o DataFrame com as classes balanceadas
df_resampled = pd.DataFrame(X_resampled, columns=[f'feature_{i}' for i in range(X_resampled.shape[1])])
df_resampled['sentimento_codificado'] = y_resampled

# Verificar a contagem de linhas após o balanceamento
print("\nContagem de linhas por sentimento_codificado após o balanceamento:")
print(df_resampled['sentimento_codificado'].value_counts())

# Dividir os dados em treino (70%) e teste (30%)
X_train, X_test, y_train, y_test = train_test_split(df_resampled.drop(columns=['sentimento_codificado']), df_resampled['sentimento_codificado'], test_size=0.3, random_state=42)

# Reconstruir dataframes completos
train_data = X_train.copy()
train_data['sentimento_codificado'] = y_train
test_data = X_test.copy()
test_data['sentimento_codificado'] = y_test

# Verificar se os nomes das colunas estão corretos
print("Colunas do conjunto de treino:", train_data.columns)
print("Colunas do conjunto de teste:", test_data.columns)

# Salvar os conjuntos de dados como arquivos Parquet usando pandas diretamente
train_data.to_parquet(config['directories']['processed_data'] + 'train_data.parquet', index=False)
test_data.to_parquet(config['directories']['processed_data'] + 'test_data.parquet', index=False)

# Atualizar o arquivo de configuração
config['files']['train_data_parquet'] = 'train_data.parquet'
config['files']['test_data_parquet'] = 'test_data.parquet'

with open(config_path, 'w') as file:
    yaml.safe_dump(config, file)

print("Divisão de dados e balanceamento concluídos com sucesso.")

# Treinamento e Avaliação de Modelos

# Carregar o dataset balanceado
df = pd.read_parquet(config['directories']['processed_data'] + 'train_data.parquet')

# Verificar as colunas disponíveis no DataFrame
print("Colunas disponíveis no DataFrame:")
print(df.columns)

# Selecionar features e target (ajustar conforme as colunas disponíveis)
features = [col for col in df.columns if col.startswith('feature')]
X = df[features]
y = df['sentimento_codificado']

# Dividir os dados em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Lista de modelos para treinar e avaliar
modelos = {
    'Logistic Regression': LogisticRegression(max_iter=1000),
    'Random Forest': RandomForestClassifier(),
    'Gradient Boosting': GradientBoostingClassifier(),
    'K-Nearest Neighbors': KNeighborsClassifier(),
    'Decision Tree': DecisionTreeClassifier(),
    'Naive Bayes': MultinomialNB(),
    'AdaBoost': AdaBoostClassifier(),
    'XGBoost': xgb.XGBClassifier(),
    'LightGBM': lgb.LGBMClassifier(),
    'Extra Trees': ExtraTreesClassifier(),
    'SGD Classifier': SGDClassifier(),
    'Support Vector Machine': SVC()
}

parametros = {
    'Logistic Regression': {'C': [0.1, 1, 10]},
    'Random Forest': {'n_estimators': [100, 200], 'max_depth': [10, 20]},
    'Gradient Boosting': {'n_estimators': [100, 200], 'learning_rate': [0.01, 0.1]},
    'K-Nearest Neighbors': {'n_neighbors': [3, 5, 7]},
    'Decision Tree': {'max_depth': [10, 20, None]},
    'Naive Bayes': {},
    'AdaBoost': {'n_estimators': [50, 100]},
    'XGBoost': {'n_estimators': [100, 200], 'learning_rate': [0.01, 0.1]},
    'LightGBM': {'n_estimators': [100, 200], 'learning_rate': [0.01, 0.1]},
    'Extra Trees': {'n_estimators': [100, 200], 'max_depth': [10, 20]},
    'SGD Classifier': {'alpha': [0.0001, 0.001, 0.01]},
    'Support Vector Machine': {'C': [0.1, 1, 10], 'kernel': ['linear', 'rbf']}
}

resultados = {}

# Treinar e avaliar cada modelo com ajuste de hiperparâmetros
for nome, modelo in modelos.items():
    print(f"Treinando e avaliando {nome}...")
    clf = GridSearchCV(modelo, parametros[nome], scoring='accuracy', cv=5)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    
    print(f"Relatório de Classificação para {nome}:")
    print(classification_report(y_test, y_pred, zero_division=0))
    
    print(f"Matriz de Confusão para {nome}:")
    print(confusion_matrix(y_test, y_pred))
    
    # Armazenar os resultados para comparação
    resultados[nome] = classification_report(y_test, y_pred, output_dict=True)['accuracy']

# Identificar o melhor modelo
melhor_modelo = max(resultados, key=resultados.get)
print(f"\nO melhor modelo é {melhor_modelo} com uma precisão de {resultados[melhor_modelo]:.2f}")

print("Classificação e avaliação de todos os modelos concluídas com sucesso.")
