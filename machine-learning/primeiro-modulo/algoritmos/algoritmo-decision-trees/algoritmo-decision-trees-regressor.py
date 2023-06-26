# Machine Learning: Decision Trees Regressor - Exercicio

# Documentação: https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeRegressor.html

#Instale o graphviz: https://graphviz.org/download/
#Instale no ANACONDA NAVIGATOR o pacote: graphviz

# Execute os blocos de códigos no Jupyter Notebook, conforme indicação: 

## Bloco1
import pandas as pd
arquivo = pd.read_csv('D:/Github/data-science/machine-learning/algoritmos/algoritmo-decision-trees/admission_predict.csv')

## Bloco2
arquivo.head()

## Bloco3
arquivo.drop('Serial No.', axis=1, inplace=True)

## Bloco4
#Separando as variáveis entre preditoras e variável target
y = arquivo['Chance of Admit ']
x = arquivo.drop('Chance of Admit ', axis = 1)

## Bloco5
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import GridSearchCV

#Definindo os valores que serão testados em DecisionTree:
minimos_split = np.array([2,3,4,5,6,7])
maximo_nivel = np.array([3,4,5,6,7,9,11])
algoritmo = ['squared_error', 'friedman_mse', 'absolute_error', 'poisson']
valores_grid = {'min_samples_split': minimos_split, 'max_depth': maximo_nivel, 'criterion': algoritmo}

#Criação do modelo:
modelo = DecisionTreeRegressor()

#Criando os grids:
gridDecisionTree = GridSearchCV(estimator = modelo, param_grid = valores_grid, cv=5)
gridDecisionTree.fit(x,y)

#Imprimindo os melhores parâmetros:
print ("Mínimo split: ", gridDecisionTree.best_estimator_.min_samples_split)
print ("Máximo profundidade: ", gridDecisionTree.best_estimator_.max_depth)
print ("Algotitmo escolhido: ", gridDecisionTree.best_estimator_.criterion)
print ("Coef. R2: ", gridDecisionTree.best_score_)
