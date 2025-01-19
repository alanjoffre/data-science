# Machine Learning: Pré-processamento: Dados Missing (Dados Faltantes)

# Dataset: Consumo e energia de prédios

# Execute os blocos de códigos no Jupyter Notebook, conforme indicação: 

## Bloco1
import pandas as pd
pd.set_option('display.max_columns', 42)
dados = pd.read_csv('D:/Ti/Cursos/Vscode/github/machine-learning/pre-processamento/2015-building-energy-benchmarking.csv')

## Bloco2
dados.head()

## Bloco3
faltantes = dados.isnull().sum()
#Fórmula cálculo: dados faltantes
faltantes_percentual = (dados.isnull().sum() / len(dados['OSEBuildingID'])) * 100
print(faltantes_percentual)

## Bloco4
#fillna - preenche os dados faltantes
dados['ENERGYSTARScore'] = dados['ENERGYSTARScore'].fillna(dados['ENERGYSTARScore'].median()) 

## Bloco5
faltantes_percentual = (dados.isnull().sum() / len(dados['OSEBuildingID'])) * 100
print(faltantes_percentual)



