import pandas as pd
import numpy as np
from tqdm import tqdm
import time

# Iniciar o cronômetro
start_time = time.time()

# Caminho do dataset
dataset_path = "D:/Github/data-science/projetos/logistica_transporte/1_previsao_gorjeta_regressao/data/raw/dataset_preprocessado.parquet"

# Importar o dataset
df = pd.read_parquet(dataset_path)

# Informar o número de linhas no dataset carregado
print(f"Número de linhas no dataset: {df.shape[0]}")

# Listar todas as colunas do dataset
print("Colunas do dataset:", df.columns.tolist())

# Imprimir os tipos de dados de cada coluna
print("Tipos de dados de cada coluna:")
print(df.dtypes)

# Converter colunas de datas para o formato correto
df['data_inicio'] = pd.to_datetime(df['data_inicio'])
df['data_final'] = pd.to_datetime(df['data_final'])

# Criar novas variáveis baseadas em datas
df['ano_inicio'] = df['data_inicio'].dt.year
df['mes_inicio'] = df['data_inicio'].dt.month
df['dia_inicio'] = df['data_inicio'].dt.day
df['ano_final'] = df['data_final'].dt.year
df['mes_final'] = df['data_final'].dt.month
df['dia_final'] = df['data_final'].dt.day

# Exibir as primeiras linhas do DataFrame para verificação
print("Primeiras linhas do DataFrame:")
print(df.head())

# Informar quais colunas possuem valores ausentes, nulos ou NaNs
colunas_nulas = df.columns[df.isnull().any()]
print("Colunas com valores ausentes, nulos ou NaNs:")
print(colunas_nulas)

# Exibir os resultados detalhados
print("Detalhes dos valores ausentes:")
print(df[colunas_nulas].isnull().sum())

# Preencher valores numéricos ausentes com a média
numerical_cols = ['segundos_da_viagem', 'milhas_da_viagem', 'area_comunitaria_do_embarque', 
                  'area_comunitaria_do_desembarque', 'tarifa', 'gorjeta', 'cobrancas_adicionais', 
                  'total_da_viagem', 'latitude_do_centroide_do_embarque', 'longitude_do_centroide_do_embarque', 
                  'latitude_do_centroide_do_desembarque', 'longitude_do_centroide_do_desembarque']

for col in tqdm(numerical_cols, desc="Preenchendo valores numéricos"):
    df[col].fillna(df[col].mean(), inplace=True)

# Preencher valores categóricos ausentes com a moda
categorical_cols = ['local_do_centroide_do_embarque', 'local_do_centroide_do_desembarque']

for col in tqdm(categorical_cols, desc="Preenchendo valores categóricos"):
    df[col].fillna(df[col].mode()[0], inplace=True)

# Preencher valores ausentes nas colunas `trato_do_censo_do_embarque` e `trato_do_censo_do_desembarque` com a mediana
df['trato_do_censo_do_embarque'].fillna(df['trato_do_censo_do_embarque'].median(), inplace=True)
df['trato_do_censo_do_desembarque'].fillna(df['trato_do_censo_do_desembarque'].median(), inplace=True)

# Verificar novamente os valores ausentes
print("Valores ausentes após o tratamento:")
print(df.isnull().sum())

# Obter uma amostra de 3% do dataset
df_sample = df.sample(frac=0.03, random_state=42)

# Informar o número de linhas na amostra
print(f"Número de linhas na amostra: {df_sample.shape[0]}")

import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from tqdm import tqdm

# Criação de novas features com base nas datas
df_sample['hora_dia_inicio'] = df_sample['data_inicio'].dt.hour
df_sample['dia_semana_inicio'] = df_sample['data_inicio'].dt.dayofweek

# Definir a variável alvo (por exemplo, 'gorjeta') e as variáveis de entrada
X = df_sample.drop(columns=['gorjeta', 'data_inicio', 'data_final'])
y = df_sample['gorjeta']

# Listar as colunas categóricas e numéricas
categorical_features = ['local_do_centroide_do_embarque', 'local_do_centroide_do_desembarque']
numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()

# Criação do pré-processador
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(), categorical_features)
    ])

# Criação do Pipeline
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', RandomForestRegressor(n_estimators=100, random_state=42))
])

# Treinamento do modelo com barra de progresso
with tqdm(total=100, desc="Treinando o modelo") as pbar:
    model.fit(X, y)
    pbar.update(100)

# Extração das importâncias das features
feature_importances = model.named_steps['model'].feature_importances_
feature_names = numeric_features + list(model.named_steps['preprocessor'].transformers_[1][1].get_feature_names_out(categorical_features))

# Criação de um DataFrame com as importâncias
feature_importance_df = pd.DataFrame({
    'feature': feature_names,
    'importance': feature_importances
}).sort_values(by='importance', ascending=False)

# Exibição das melhores features
print("Importância das Features:")
print(feature_importance_df)

# Salvar as importâncias das features
processed_dir = "D:/Github/data-science/projetos/logistica_transporte/1_previsao_gorjeta_regressao/data/processed/"
feature_importance_df.to_csv(processed_dir + "feature_importances.csv", index=False)
feature_importance_df.to_parquet(processed_dir + "feature_importances.parquet", index=False)

print("Importâncias das features salvas com sucesso.")

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Caminho do dataset processado
processed_path = "D:/Github/data-science/projetos/logistica_transporte/1_previsao_gorjeta_regressao/data/processed/dataset_processado.parquet"

# Importar o dataset processado
df = pd.read_parquet(processed_path)

# Gráfico de dispersão para ver a relação entre a distância da viagem e o valor da gorjeta
plt.figure(figsize=(10, 6))
sns.scatterplot(x=df['milhas_da_viagem'], y=df['gorjeta'])
plt.title("Relação entre Milhas da Viagem e Gorjeta")
plt.xlabel("Milhas da Viagem")
plt.ylabel("Gorjeta")

# Salvar o gráfico
figures_dir = "D:/Github/data-science/projetos/logistica_transporte/1_previsao_gorjeta_regressao/reports/figures/"
plt.savefig(figures_dir + "relacao_milhas_gorjeta.png")
plt.close()

# Calcular correlações entre variáveis
corr_matrix = df.corr()

# Visualizar a matriz de correlação usando heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.title("Heatmap de Correlação entre Variáveis")

# Salvar o heatmap
plt.savefig(figures_dir + "heatmap_correlacao.png")
plt.close()

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import os
import time

# Caminho do dataset processado
processed_path = "D:/Github/data-science/projetos/logistica_transporte/1_previsao_gorjeta_regressao/data/processed/dataset_processado.parquet"

# Importar o dataset processado
df = pd.read_parquet(processed_path)

# Definir a variável alvo (por exemplo, 'gorjeta') e as variáveis de entrada
X = df.drop(columns=['gorjeta'])
y = df['gorjeta']

# Dividir o dataset em conjuntos de treinamento (80%) e teste (20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Tamanho do conjunto de treinamento: {X_train.shape[0]} linhas")
print(f"Tamanho do conjunto de teste: {X_test.shape[0]} linhas")

# Caminho para salvar os conjuntos de treinamento e teste
train_test_dir = "D:/Github/data-science/projetos/logistica_transporte/1_previsao_gorjeta_regressao/data/processed/train_test/"
os.makedirs(train_test_dir, exist_ok=True)

# Salvar os conjuntos de treinamento e teste
X_train.to_csv(train_test_dir + "X_train.csv", index=False)
X_test.to_csv(train_test_dir + "X_test.csv", index=False)
y_train.to_csv(train_test_dir + "y_train.csv", index=False)
y_test.to_csv(train_test_dir + "y_test.csv", index=False)

# Parar o cronômetro
end_time = time.time()
total_time = end_time - start_time

# Converter o tempo total para horas, minutos e segundos
hours, remainder = divmod(total_time, 3600)
minutes, seconds = divmod(remainder, 60)

print(f"Tempo total de processamento: {int(hours)} horas, {int(minutes)} minutos e {int(seconds)} segundos")
