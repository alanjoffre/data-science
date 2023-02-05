# Machine Learning: Logistic Regression - Exercicio

# Documentação: https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_breast_cancer.html

# Execute os blocos de códigos no Jupyter Notebook, conforme indicação: 

## Bloco1
from sklearn.datasets import load_breast_cancer
import pandas as pd
import warnings
pd.set_option('display.max_columns', 30)
warnings.filterwarnings('ignore')
dados = load_breast_cancer()
x = pd.DataFrame(dados.data, columns=[dados.feature_names])
y = pd.Series(dados.target)

## Bloco2
print(dados)

## Bloco3
type(dados)

## Bloco4
x.head()

## Bloco5
y.head(30)

## Bloco6
print(x.shape, y.shape)

## Bloco7
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV

#Definindo os valores que serão testado em LogisticRegression
valores_C = np.array([0.01, 0.1, 0.5, 1, 2, 3, 5, 10, 20, 50, 100])
regularizacao = ['l1', 'l2']
valores_grid = {'C':valores_C, 'penalty':regularizacao}

#Criando o modelo:
modelo = LogisticRegression(solver='lbfgs', max_iter=10000)

#Criando os grids:
grid_regressao_logistica = GridSearchCV(estimator = modelo, param_grid = valores_grid, cv=5)
grid_regressao_logistica.fit(x,y)

#Imprimindo a melhor acurácia e os melhores parâmetros:
print('Melhor acurácia: ', grid_regressao_logistica.best_score_)
print('Parâmetro C: ', grid_regressao_logistica.best_estimator_.C)
print('Regularização: ', grid_regressao_logistica.best_estimator_.penalty)

## Bloco8
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV

#Definindo os valores que serão testado em LogisticRegression
valores_C = np.array([95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105])
regularizacao = ['l1', 'l2']
valores_grid = {'C':valores_C, 'penalty':regularizacao}

#Criando o modelo:
modelo = LogisticRegression( penalty= 'l2',solver='lbfgs', max_iter=10000)

#Criando os grids:
grid_regressao_logistica = GridSearchCV(estimator = modelo, param_grid = valores_grid, cv=5)
grid_regressao_logistica.fit(x,y)

#Imprimindo a melhor acurácia e os melhores parâmetros:
print('Melhor acurácia: ', grid_regressao_logistica.best_score_)
print('Parâmetro C: ', grid_regressao_logistica.best_estimator_.C)
print('Regularização: ', grid_regressao_logistica.best_estimator_.penalty)









