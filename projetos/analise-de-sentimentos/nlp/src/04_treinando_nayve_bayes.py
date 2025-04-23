# Importe as bibliotecas
import pandas as pd
import dask.dataframe as dd
from sklearn.model_selection import train_test_split, RandomizedSearchCV, StratifiedKFold, cross_val_predict
from sklearn.metrics import classification_report
from sklearn.feature_extraction.text import TfidfVectorizer
from imblearn.over_sampling import SMOTE
from sklearn.naive_bayes import MultinomialNB
import numpy as np
import joblib
import json
import time

# Registrar a hora inicial do processamento
start_time = time.time()

# Carregar o dataset com Dask
file_path = r'D:\Github\data-science\projetos\analise-de-sentimentos\nlp\data\processed\final_data.parquet'
df = dd.read_parquet(file_path)

# Definir o número de linhas por bloco
block_size = 1000

# Inicializar o vetor de TfidfVectorizer
vectorizer = TfidfVectorizer(max_features=20000)

# Inicializar o modelo Naive Bayes com partial_fit
nb_model = MultinomialNB()

# Listas para armazenar métricas e dados de cada bloco
results_train = []
results_test = []
X_blocks = []
y_blocks = []

# Ajustar o vetor de TF-IDF em todo o conjunto de dados
X = df['tweet'].compute()
vectorizer.fit(X)

# Salvar o vetor de TF-IDF ajustado
preprocessor_path = r'D:\Github\data-science\projetos\analise-de-sentimentos\nlp\preprocessors\tfidf_vectorizer.pkl'
joblib.dump(vectorizer, preprocessor_path)

# Processar cada bloco de 1000 linhas
for i in range(0, df.shape[0].compute(), block_size):
    df_bloco = df.loc[i:i+block_size-1].compute()
    X = df_bloco['tweet']
    y = df_bloco['sentimento_codificado']

    # Vetorização de textos usando o vetor de TF-IDF ajustado
    X_vec = vectorizer.transform(X).toarray()

    # Lidar com dados desbalanceados utilizando SMOTE
    smote = SMOTE(sampling_strategy='minority', random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X_vec, y)

    # Guardar os dados para posterior uso na busca de hiperparâmetros
    X_blocks.append(X_resampled)
    y_blocks.append(y_resampled)

    # Dividir o bloco em conjunto de treinamento e teste
    X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.3, random_state=42)

    # Treinar o modelo incrementalmente
    nb_model.partial_fit(X_train, y_train, classes=np.unique(y_resampled))

    # Avaliar o modelo no conjunto de treinamento
    y_train_pred = nb_model.predict(X_train)
    train_report = classification_report(y_train, y_train_pred, zero_division=0, output_dict=True)
    results_train.append(train_report)

    # Avaliar o modelo no conjunto de teste
    y_test_pred = nb_model.predict(X_test)
    test_report = classification_report(y_test, y_test_pred, zero_division=0, output_dict=True)
    results_test.append(test_report)

# Função para agrupar os resultados de múltiplos blocos
def agregar_resultados(resultados):
    agregados = {}
    for key in resultados[0].keys():
        valores = [result[key] for result in resultados]
        if isinstance(valores[0], dict):
            agregados[key] = agregar_resultados(valores)
        else:
            agregados[key] = np.mean(valores)
    return agregados

# Agregar resultados de todos os blocos
final_train_report = agregar_resultados(results_train)
final_test_report = agregar_resultados(results_test)

# Exibir o relatório de classificação final
print("Relatório de Classificação no conjunto de treinamento:")
print(json.dumps(final_train_report, indent=4))
print("Relatório de Classificação no conjunto de teste:")
print(json.dumps(final_test_report, indent=4))

# Encontrar os melhores hiperparâmetros usando RandomizedSearchCV
param_distributions = {
    'alpha': np.linspace(0.01, 1.0, 50)  # Intervalo de valores para o parâmetro alpha
}

search = RandomizedSearchCV(
    MultinomialNB(),
    param_distributions=param_distributions,
    n_iter=50,
    cv=5,
    scoring='f1_weighted',
    random_state=42,
    n_jobs=-1
)

# Selecionar uma amostra menor para a busca de hiperparâmetros devido ao grande volume de dados
sample_blocks = min(2, len(X_blocks))  # Usar no máximo 2 blocos ou a quantidade disponível
X_sample = np.vstack([X_blocks[i] for i in range(sample_blocks)])
y_sample = np.hstack([y_blocks[i] for i in range(sample_blocks)])

# Ajustar a busca de hiperparâmetros na amostra selecionada
search.fit(X_sample, y_sample)
nb_model = search.best_estimator_
best_params = search.best_params_
print(f"\nMelhores hiperparâmetros para Naive Bayes: {best_params}")

# Treinar novamente o modelo com os melhores hiperparâmetros
for i in range(len(X_blocks)):
    X_resampled = X_blocks[i]
    y_resampled = y_blocks[i]
    nb_model.partial_fit(X_resampled, y_resampled, classes=np.unique(y_resampled))

# Aplicar validação cruzada em uma amostra do conjunto de dados
cross_val_sample_size = min(2, len(X_blocks))  # Usar no máximo 2 blocos para validação cruzada
X_sample_val = np.vstack([X_blocks[i] for i in range(cross_val_sample_size)])
y_sample_val = np.hstack([y_blocks[i] for i in range(cross_val_sample_size)])
skf = StratifiedKFold(n_splits=5, random_state=42, shuffle=True)
cross_val_results = cross_val_predict(nb_model, X_sample_val, y_sample_val, cv=skf)
print("Relatório de Classificação na amostra com validação cruzada:\n" + classification_report(y_sample_val, cross_val_results, zero_division=0))

# Salvar o modelo treinado após a validação cruzada
model_path = r'D:\Github\data-science\projetos\analise-de-sentimentos\nlp\models\best_model_nb.pkl'
joblib.dump(nb_model, model_path)

# Salvar os hiperparâmetros
params_path = r'D:\Github\data-science\projetos\analise-de-sentimentos\nlp\reports\best_model_params_nb.json'
with open(params_path, 'w') as file:
    json.dump(best_params, file)

# Registrar a hora final do processamento
end_time = time.time()

# Calcular a duração do processamento
processing_duration = end_time - start_time

# Converter duração para horas, minutos e segundos
hours, remainder = divmod(processing_duration, 3600)
minutes, seconds = divmod(remainder, 60)
print(f"Tempo total de processamento: {int(hours)} horas, {int(minutes)} minutos e {seconds:.2f} segundos")
