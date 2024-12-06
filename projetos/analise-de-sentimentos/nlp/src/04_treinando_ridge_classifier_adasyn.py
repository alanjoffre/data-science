import pandas as pd
import dask.dataframe as dd
from textblob import TextBlob
import numpy as np  # Certifique-se de importar a biblioteca NumPy

# Função de Feature Engineering
def feature_engineering(df):
    df['word_count'] = df['tweet'].apply(lambda x: len(str(x).split()))
    df['char_count'] = df['tweet'].apply(lambda x: len(str(x)))
    df['avg_word_length'] = df['tweet'].apply(lambda x: np.mean([len(w) for w in str(x).split()]))
    df['sentiment'] = df['tweet'].apply(lambda x: TextBlob(str(x)).sentiment.polarity)
    return df

# Carregar o dataset em pedaços utilizando Dask
file_path = r'D:\Github\data-science\projetos\analise-de-sentimentos\nlp\data\processed\final_data.parquet'
chunksize = 500  # Ajustar para 500 linhas para reduzir uso de memória
df_dask = dd.read_parquet(file_path, chunksize=chunksize)

# Processar cada pedaço
df_processed = []

for chunk in df_dask.partitions:
    chunk_df = chunk.compute()  # Converter o pedaço em um dataframe Pandas
    
    # Aplicar Feature Engineering
    chunk_df = feature_engineering(chunk_df)
    df_processed.append(chunk_df)

# Concatenar todos os pedaços processados
df_final = pd.concat(df_processed).reset_index(drop=True)

from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV, GridSearchCV, train_test_split, cross_val_predict
from sklearn.linear_model import RidgeClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from imblearn.over_sampling import ADASYN
import numpy as np
import joblib
import json

# Função para treinar e ajustar hiperparâmetros do modelo
def train_and_tune_model(X_train, y_train):
    # Treinar o modelo RidgeClassifier com RandomizedSearchCV para busca ampla dos hiperparâmetros
    param_distributions_ridge = {
        'alpha': np.linspace(0.1, 2, 100),  # Aumentando o intervalo de pesquisa para alpha
        'solver': ['svd', 'sag', 'saga', 'lsqr', 'cholesky', 'sparse_cg']  # Solvers suportados
    }

    ridge = RidgeClassifier()
    random_search_ridge = RandomizedSearchCV(estimator=ridge, param_distributions=param_distributions_ridge, cv=StratifiedKFold(n_splits=5), n_iter=100, scoring='f1_weighted', n_jobs=-1, verbose=2)  # Aumentando o número de iterações e splits
    random_search_ridge.fit(X_train, y_train)

    # Melhor modelo encontrado na RandomizedSearchCV
    best_ridge_random = random_search_ridge.best_estimator_
    best_params_random = random_search_ridge.best_params_
    print("Melhores Hiperparâmetros - RandomizedSearchCV - RidgeClassifier:", best_params_random)

    # Refinar com GridSearchCV ao redor dos melhores parâmetros encontrados
    param_grid_ridge_refine = {
        'alpha': [best_params_random['alpha'] * 0.9, best_params_random['alpha'], best_params_random['alpha'] * 1.1],
        'solver': [best_params_random['solver']]
    }

    grid_search_ridge = GridSearchCV(estimator=ridge, param_grid=param_grid_ridge_refine, cv=StratifiedKFold(n_splits=5), scoring='f1_weighted', n_jobs=-1, verbose=2)
    grid_search_ridge.fit(X_train, y_train)

    # Melhor modelo encontrado na GridSearchCV
    best_ridge = grid_search_ridge.best_estimator_
    print("Melhores Hiperparâmetros - GridSearchCV - RidgeClassifier:", grid_search_ridge.best_params_)

    return best_ridge, grid_search_ridge.best_params_

# Features e target
X = df_final[['tweet', 'word_count', 'char_count', 'avg_word_length', 'sentiment']]
y = df_final['sentimento_codificado']

# Vetorização de textos com TF-IDF
vectorizer = TfidfVectorizer(max_features=5000)  # Limitar o número de features para reduzir uso de memória
X_tfidf = vectorizer.fit_transform(X['tweet']).toarray()
X_features = np.hstack((X_tfidf, X[['word_count', 'char_count', 'avg_word_length', 'sentiment']].values))

# Dividir o dataset em conjunto de treinamento e teste (70/30)
X_train, X_test, y_train, y_test = train_test_split(X_features, y, test_size=0.3, random_state=42)

# Aplicar ADASYN ao invés de SMOTE
adasyn = ADASYN(sampling_strategy='minority')
X_train_resampled, y_train_resampled = adasyn.fit_resample(X_train, y_train)

# Treinar e ajustar o modelo
final_model, best_params = train_and_tune_model(X_train_resampled, y_train_resampled)

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

from sklearn.metrics import classification_report
import joblib
import numpy as np

# Carregar o vetor de TfidfVectorizer e o modelo treinado
preprocessor_path = r'D:\Github\data-science\projetos\analise-de-sentimentos\nlp\preprocessors\tfidf_vectorizer.pkl'
model_path = r'D:\Github\data-science\projetos\analise-de-sentimentos\nlp\models\best_ridge_model.pkl'
vectorizer = joblib.load(preprocessor_path)
final_model = joblib.load(model_path)

# Features e target
X = df_final[['tweet', 'word_count', 'char_count', 'avg_word_length', 'sentiment']]
y = df_final['sentimento_codificado']

# Dividir o dataset em conjunto de treinamento e teste (70/30)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Transformar a coluna de texto para TF-IDF
X_test_vec = vectorizer.transform(X_test['tweet'])

# Adicionar as outras features
X_test_features = np.hstack((X_test_vec.toarray(), X_test[['word_count', 'char_count', 'avg_word_length', 'sentiment']].values))

# Gerar previsões no conjunto de teste
y_pred_test_ridge = final_model.predict(X_test_features)

# Verificar consistência dos dados
assert len(y_test) == len(y_pred_test_ridge), "Inconsistência no número de amostras entre y_test e y_pred_test_ridge"

# Gerar relatório de classificação
print("Relatório de Classificação - RidgeClassifier Otimizado:\n" + classification_report(y_test, y_pred_test_ridge, zero_division=0))

# Realizar validação cruzada
X_train_features = np.hstack((vectorizer.transform(X_train['tweet']).toarray(), X_train[['word_count', 'char_count', 'avg_word_length', 'sentiment']].values))
y_pred_cv_ridge = cross_val_predict(final_model, X_train_features, y_train, cv=StratifiedKFold(n_splits=5))
print("Relatório de Classificação - Validação Cruzada RidgeClassifier:\n" + classification_report(y_train, y_pred_cv_ridge, zero_division=0))

from sklearn.metrics import roc_curve, auc, classification_report
import matplotlib.pyplot as plt
import numpy as np
import joblib

# Carregar o vetor de TfidfVectorizer e o modelo treinado
preprocessor_path = r'D:\Github\data-science\projetos\analise-de-sentimentos\nlp\preprocessors\tfidf_vectorizer.pkl'
model_path = r'D:\Github\data-science\projetos\analise-de-sentimentos\nlp\models\best_ridge_model.pkl'
vectorizer = joblib.load(preprocessor_path)
final_model = joblib.load(model_path)

# Features e target
X = df_final[['tweet', 'word_count', 'char_count', 'avg_word_length', 'sentiment']]
y = df_final['sentimento_codificado']

# Dividir o dataset em conjunto de treinamento e teste (70/30)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Transformar a coluna de texto para TF-IDF
X_test_vec = vectorizer.transform(X_test['tweet'])

# Adicionar as outras features
X_test_features = np.hstack((X_test_vec.toarray(), X_test[['word_count', 'char_count', 'avg_word_length', 'sentiment']].values))

# Encontrar o melhor threshold usando a curva ROC para cada classe individualmente
fpr = {}
tpr = {}
roc_auc = {}
optimal_thresholds = {}
y_scores_ridge = final_model.decision_function(X_test_features)

for i in range(len(np.unique(y_test))):
    y_test_binary = (y_test == i).astype(int)
    y_scores_ridge_binary = y_scores_ridge[:, i]
    
    fpr[i], tpr[i], roc_thresholds_ridge = roc_curve(y_test_binary, y_scores_ridge_binary)
    roc_auc[i] = auc(fpr[i], tpr[i])

    optimal_idx_ridge = np.argmax(tpr[i] - fpr[i])
    optimal_thresholds[i] = roc_thresholds_ridge[optimal_idx_ridge]
    print(f"Melhor Threshold pela Curva ROC - RidgeClassifier para a classe {i}: {optimal_thresholds[i]}")

    # Aplicar o melhor threshold
    y_pred_best_threshold_ridge = (y_scores_ridge_binary >= optimal_thresholds[i]).astype(int)
    y_test_multi = np.copy(y_test)
    y_test_multi[y_test != i] = -1
    y_pred_multi = np.copy(y_pred_test_ridge)
    y_pred_multi[y_pred_test_ridge != i] = -1
    print(f"Relatório de Classificação - Melhor Threshold pela Curva ROC - RidgeClassifier para a classe {i}:\n" + classification_report(y_test_multi, y_pred_multi, zero_division=0))

# Salvar o modelo treinado com o melhor threshold
joblib.dump(final_model, r'D:\Github\data-science\projetos\analise-de-sentimentos\nlp\models\ridge_model_threshold.pkl')

# Plotar as curvas ROC e salvar os gráficos
for i in range(len(np.unique(y_test))):
    plt.figure()
    plt.plot(fpr[i], tpr[i], color='darkorange', lw=2, label=f'Curva ROC (área = {roc_auc[i]:0.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.scatter(fpr[i][optimal_idx_ridge], tpr[i][optimal_idx_ridge], marker='o', color='red', label='Melhor Threshold')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Taxa de Falsos Positivos')
    plt.ylabel('Taxa de Verdadeiros Positivos')
    plt.title(f'Curva ROC - RidgeClassifier para a classe {i}')
    plt.legend(loc="lower right")

    # Salvar o gráfico
    plt.savefig(r'D:\Github\data-science\projetos\analise-de-sentimentos\nlp\reports\figures\roc_curve_ridge_class_{}.png'.format(i))
    plt.close()

print("Avaliação e visualização dos resultados concluídas!")
