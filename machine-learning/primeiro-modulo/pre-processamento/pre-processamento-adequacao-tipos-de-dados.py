# Machine Learning: Pré-processamento: Adequação dos tipos de dados(int, float, string)

# Dataset: Consumo e energia de prédios

# Execute os blocos de códigos no Jupyter Notebook, conforme indicação: 

## Bloco1
import pandas as pd
pd.set_option('display.max_columns', 42)
dados = pd.read_csv('D:/Ti/Cursos/Vscode/github/machine-learning/pre-processamento/2015-building-energy-benchmarking.csv')

## Bloco2
dados.head()

## Bloco3
dados.dtypes # Descobrir qual o tipo de dado de cada variável

## Bloco4
dados['DataYear'] = dados['DataYear'].astype(object)

## Bloco5
dados.dtypes # Descobrir qual o tipo de dado de cada variável

# IMPORTANTE: Dependendo do modelo de Machine Learning aplicado, identidicará  somente int(número)
