import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import const
from utils import *

df = fetch_data_from_db(const.consulta_sql)

df['idade'] = df['idade'].astype(int)
df['valorsolicitado'] = df['valorsolicitado'].astype(float)
df['valortotalbem'] = df['valortotalbem'].astype(float)

variaveis_categoricas = ['profissao', 'tiporesidencia', 
                         'escolaridade', 'score', 'estadocivil', 'produto']
variaveis_numericas = ['tempoprofissao', 'renda', 'idade',
                        'dependentes', 'valorsolicitado', 'valortotalbem']

for coluna in variaveis_categoricas:
    df[coluna].value_counts().plot(kind='bar', figsize=(10, 6))
    plt.title(f'Distribuição de {coluna}')
    plt.ylabel('Contagem')
    plt.xlabel(coluna)
    plt.xticks(rotation=45)
    plt.show()

for coluna in variaveis_numericas:
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=df, x=coluna)
    plt.title(f'Boxplot de {coluna}')
    plt.show()

    df[coluna].hist(bins=20, figsize=(10, 6))
    plt.title(f'Histograma de {coluna}')
    plt.xlabel(coluna)
    plt.ylabel('Frequência')
    plt.show()

    print(f'Resumo estatístico de :\n', df[coluna].describe(), '\n')


nulos_por_coluna = df.isnull().sum()
print(nulos_por_coluna)