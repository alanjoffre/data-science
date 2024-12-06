import pandas as pd
import dask.dataframe as dd
from textblob import TextBlob
import numpy as np
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV, GridSearchCV, train_test_split, cross_val_predict
from sklearn.linear_model import RidgeClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from imblearn.over_sampling import SMOTE
from sklearn.metrics import classification_report, f1_score
import joblib
import json

# Função de Feature Engineering
def feature_engineering(df):
    df['word_count'] = df['tweet'].apply(lambda x: len(str(x).split()))
    df['char_count'] = df['tweet'].apply(lambda x: len(str(x)))
    df['avg_word_length'] = df['tweet'].apply(lambda x: np.mean([len(w) for w in str(x).split()]))
    df['sentiment'] = df['tweet'].apply(lambda x: TextBlob(str(x)).sentiment.polarity)
    return df

# Carregar o dataset em pedaços utilizando Dask
file_path = r'D:\Github\data-science\projetos\analise-de-sentimentos\nlp\data\processed\final_data.parquet'
chunksize = 500
df_dask = dd.read_parquet(file_path, chunksize=chunksize)

# Processar cada pedaço
df_processed = []
for chunk in df_dask.partitions:
    chunk_df = chunk.compute()
    chunk_df = feature_engineering(chunk_df)
    df_processed.append(chunk_df)

# Concatenar todos os pedaços processados
df_final = pd.concat(df_processed).reset_index(drop=True)

# Features e target
X = df_final[['tweet', 'word_count', 'char_count', 'avg_word_length', 'sentiment']]
y = df_final['sentimento_codificado']

# Vetorização de textos com TF-IDF
vectorizer = TfidfVectorizer(max_features=5000)
X_tfidf = vectorizer.fit_transform(X['tweet']).toarray()
X_features = np.hstack((X_tfidf, X[['word_count', 'char_count', 'avg_word_length', 'sentiment']].values))

# Dividir o dataset em conjunto de treinamento e teste (70/30)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Transformar a coluna de texto para TF-IDF
X_train_vec = vectorizer.transform(X_train['tweet'])
X_test_vec = vectorizer.transform(X_test['tweet'])

# Adicionar as outras features
X_train_features = np.hstack((X_train_vec.toarray(), X_train[['word_count', 'char_count', 'avg_word_length', 'sentiment']].values))
X_test_features = np.hstack((X_test_vec.toarray(), X_test[['word_count', 'char_count', 'avg_word_length', 'sentiment']].values))

# Verificar a distribuição das classes no conjunto de treinamento
print("Distribuição das classes no conjunto de treinamento:")
print(pd.Series(y_train).value_counts())

# Aplicar SMOTE para balancear as classes
smote = SMOTE(sampling_strategy='minority')
X_train_resampled, y_train_resampled = smote.fit_resample(X_train_features, y_train)

# Verificar a distribuição das classes após o SMOTE
print("Distribuição das classes após SMOTE:")
print(pd.Series(y_train_resampled).value_counts())

# Função para treinar e ajustar hiperparâmetros do modelo
def train_and_tune_model(X_train, y_train):
    param_distributions_ridge = {
        'alpha': np.linspace(0.1, 2, 100),
        'solver': ['svd', 'sag', 'saga', 'lsqr', 'cholesky', 'sparse_cg']
    }

    ridge = RidgeClassifier()
    random_search_ridge = RandomizedSearchCV(estimator=ridge, param_distributions=param_distributions_ridge, cv=StratifiedKFold(n_splits=5), n_iter=100, scoring='f1_weighted', n_jobs=-1, verbose=2)
    random_search_ridge.fit(X_train, y_train)

    best_ridge_random = random_search_ridge.best_estimator_
    best_params_random = random_search_ridge.best_params_
    print("Melhores Hiperparâmetros - RandomizedSearchCV - RidgeClassifier:", best_params_random)

    param_grid_ridge_refine = {
        'alpha': [best_params_random['alpha'] * 0.9, best_params_random['alpha'], best_params_random['alpha'] * 1.1],
        'solver': [best_params_random['solver']]
    }

    grid_search_ridge = GridSearchCV(estimator=ridge, param_grid=param_grid_ridge_refine, cv=StratifiedKFold(n_splits=5), scoring='f1_weighted', n_jobs=-1, verbose=2)
    grid_search_ridge.fit(X_train, y_train)

    best_ridge = grid_search_ridge.best_estimator_
    print("Melhores Hiperparâmetros - GridSearchCV - RidgeClassifier:", grid_search_ridge.best_params_)

    return best_ridge, grid_search_ridge.best_params_

# Treinar e ajustar o modelo
final_model, best_params = train_and_tune_model(X_train_resampled, y_train_resampled)

# Função para gerar e imprimir relatórios de classificação
def generate_classification_reports(model, X_train, y_train, X_test, y_test):
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    print("Relatório de Classificação - Conjunto de Treinamento:")
    print(classification_report(y_train, y_train_pred))

    print("Relatório de Classificação - Conjunto de Teste:")
    print(classification_report(y_test, y_test_pred))

# Gera e imprime os relatórios de classificação
generate_classification_reports(final_model, X_train_resampled, y_train_resampled, X_test_features, y_test)

# Validação cruzada e relatório de classificação
def cross_validation_report(model, X, y):
    y_pred = cross_val_predict(model, X, y, cv=StratifiedKFold(n_splits=5))
    print("Relatório de Classificação - Validação Cruzada:")
    print(classification_report(y, y_pred))

cross_validation_report(final_model, X_features, y)

# Função para encontrar o melhor threshold e gerar o relatório de classificação
def find_best_threshold(model, X_train, y_train, X_test, y_test):
    thresholds = np.linspace(0, 1, 100)
    melhor_threshold = 0
    melhor_f1 = 0
    for threshold in thresholds:
        y_train_pred = (model.decision_function(X_train) > threshold).astype(int)
        y_test_pred = (model.decision_function(X_test) > threshold).astype(int)
        
        if len(np.unique(y_test)) > 2:  # Caso multiclasse
            y_train_pred = np.argmax(y_train_pred, axis=1)
            y_test_pred = np.argmax(y_test_pred, axis=1)
            f1 = f1_score(y_test, y_test_pred, average='weighted')
        else:  # Caso binário
            f1 = f1_score(y_test, y_test_pred, average='binary')

        if f1 > melhor_f1:
            melhor_f1 = f1
            melhor_threshold = threshold
    
    print("Melhor Threshold:", melhor_threshold)
    return melhor_threshold

# Encontrar e aplicar o melhor threshold
best_threshold = find_best_threshold(final_model, X_train_resampled, y_train_resampled, X_test_features, y_test)

# Função para aplicar o melhor threshold e gerar o relatório de classificação
def apply_best_threshold(model, X_train, y_train, X_test, y_test, threshold):
    y_train_pred = (model.decision_function(X_train) > threshold).astype(int)
    y_test_pred = (model.decision_function(X_test) > threshold).astype(int)

    if len(np.unique(y_test)) > 2:  # Caso multiclasse
        y_train_pred = np.argmax(y_train_pred, axis=1)
        y_test_pred = np.argmax(y_test_pred, axis=1)
    
    print("Relatório de Classificação - Conjunto de Treinamento com Melhor Threshold:")
    print(classification_report(y_train, y_train_pred))

    print("Relatório de Classificação - Conjunto de Teste com Melhor Threshold:")
    print(classification_report(y_test, y_test_pred))

# Aplicar o melhor threshold e gerar relatórios de classificação
apply_best_threshold(final_model, X_train_resampled, y_train_resampled, X_test_features, y_test, best_threshold)

# Salvar o vetor de TfidfVectorizer
preprocessor_path = r'D:\Github\data-science\projetos\analise-de-sentimentos\nlp\preprocessors\tfidf_vectorizer.pkl'
joblib.dump(vectorizer, preprocessor_path)

# Salvar o modelo treinado
model_path = r'D:\Github\data-science\projetos\analise-de-sentimentos\nlp\models\best_ridge_model.pkl'
joblib.dump(final_model, model_path)

# Salvar os hiperparâmetros
params_path = r'D:\Github\data-science\projetos\analise-de-sentimentos\nlp\reports\best_ridge_params.json'
with open(params_path, 'w') as file:
    json.dump(best_params, file)

print("Modelo treinado e salvo com sucesso!")

# Função para encontrar o melhor threshold e gerar o relatório de classificação
def find_best_threshold(model, X_train, y_train, X_test, y_test):
    thresholds = np.linspace(0, 1, 100)
    melhor_threshold = 0
    melhor_f1 = 0
    for threshold in thresholds:
        y_train_pred = (model.decision_function(X_train) > threshold).astype(int)
        y_test_pred = (model.decision_function(X_test) > threshold).astype(int)
        
        # Certifique-se de que tanto y_test quanto y_test_pred estejam no mesmo formato
        if len(np.unique(y_test)) > 2:  # Caso multiclasse
            y_train_pred = np.argmax(y_train_pred, axis=1)
            y_test_pred = np.argmax(y_test_pred, axis=1)
            f1 = f1_score(y_test, y_test_pred, average='weighted')
        else:  # Caso binário
            f1 = f1_score(y_test, y_test_pred, average='binary')

        if f1 > melhor_f1:
            melhor_f1 = f1
            melhor_threshold = threshold
    
    print("Melhor Threshold:", melhor_threshold)
    return melhor_threshold

# Encontrar e aplicar o melhor threshold
best_threshold = find_best_threshold(final_model, X_train_resampled, y_train_resampled, X_test_features, y_test)

# Função para aplicar o melhor threshold e gerar o relatório de classificação
def apply_best_threshold(model, X_train, y_train, X_test, y_test, threshold):
    y_train_pred = (model.decision_function(X_train) > threshold).astype(int)
    y_test_pred = (model.decision_function(X_test) > threshold).astype(int)

    # Certifique-se de que tanto y_test quanto y_test_pred estejam no mesmo formato
    if len(np.unique(y_test)) > 2:  # Caso multiclasse
        y_train_pred = np.argmax(y_train_pred, axis=1)
        y_test_pred = np.argmax(y_test_pred, axis=1)
    
    # Relatório de classificação para o conjunto de treinamento com o melhor threshold
    print("Relatório de Classificação - Conjunto de Treinamento com Melhor Threshold:")
    print(classification_report(y_train, y_train_pred))

    # Relatório de classificação para o conjunto de teste com o melhor threshold
    print("Relatório de Classificação - Conjunto de Teste com Melhor Threshold:")
    print(classification_report(y_test, y_test_pred))

# Aplicar o melhor threshold e gerar relatórios de classificação
apply_best_threshold(final_model, X_train_resampled, y_train_resampled, X_test_features, y_test, best_threshold)

print("Avaliação e visualização dos resultados concluídas!")
