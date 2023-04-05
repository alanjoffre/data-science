# Machine Learning: Feature Selection - F-Value ou F-Classif

# Importante: Trabalha com valores positivos e negativos. 

# Execute os blocos de códigos no Jupyter Notebook, conforme indicação: 

## Bloco1
from sklearn.datasets import load_iris
import pandas as pd
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
import warnings
warnings.filterwarnings('ignore')

#Carregando o dataset
iris = load_iris()

#Definindo as variaveis preditoras e a variavel target
x = pd.DataFrame(iris.data, columns=[iris.feature_names])
y = pd.Series(iris.target)

print(x.head())

## Bloco2
#selecionado duas variaveis com o maior F-Value:
algoritmo = SelectKBest(score_func=f_classif, k=2)
dados_das_melhores_preditoras = algoritmo.fit_transform(x, y)

#Resultados
print('Scores: ', algoritmo.scores_)
print('Resultado da transformação:\n', dados_das_melhores_preditoras)

## Bloco3


#Resultados




