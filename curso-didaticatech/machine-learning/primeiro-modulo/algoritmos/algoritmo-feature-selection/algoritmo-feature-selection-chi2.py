# Machine Learning: Feature Selection - CHI2

# Importante: Só trabalha com valores positivos. 

# Execute os blocos de códigos no Jupyter Notebook, conforme indicação: 

## Bloco1
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

#Definindo variáveis preditoras e target
x = [[12,2,30], [15,11,6], [16,8,90], [5,3,20], [4,14,5], [2,5,70]]
y = [1,1,1,0,0,0]

#Selecionando duas variáveis com o maior chi-quadrado:
algoritmo = SelectKBest(score_func=chi2, k=2)
dados_das_melhores_preditoras = algoritmo.fit_transform(x,y)

#Resultados
print('Score: ', algoritmo.scores_)
print('Resultado da transformação:\n', dados_das_melhores_preditoras)

## Bloco2
from sklearn.datasets import load_iris
import pandas as pd
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
import warnings
warnings.filterwarnings('ignore')

#Carregando o dataset
iris = load_iris()

#Definindo as variáveis preditoras e a variável target
x = pd.DataFrame(iris.data, columns=[iris.feature_names])
y = pd.Series(iris.target)

print(x.head())

## Bloco3
#Selecionado duas variáveis com o maior chi-quadrado:
algoritmo = SelectKBest(score_func=chi2, k=2)
dados_das_melhores_preditoras = algoritmo.fit_transform(x,y)

#Resultados
print('Scores: ', algoritmo.scores_)
print('Resultado da transformação:\n', dados_das_melhores_preditoras)



