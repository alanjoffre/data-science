# Machine Learning: Escolhendo outros tipos de Scoring

# Documentação: https://scikit-learn.org/stable/modules/model_evaluation.html

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
gridDecisionTree = GridSearchCV(estimator = modelo, param_grid = valores_grid, cv=5, scoring='neg_mean_squared_log_error')
gridDecisionTree.fit(x,y)

#Imprimindo os melhores parâmetros:
print ("Mínimo split: ", gridDecisionTree.best_estimator_.min_samples_split)
print ("Máxima profundidade: ", gridDecisionTree.best_estimator_.max_depth)
print ("Algotitmo escolhido: ", gridDecisionTree.best_estimator_.criterion)
print ("Erro médio quadrático: ", gridDecisionTree.best_score_)

## Bloco6
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold

#Separando os dados em folds
kfold = KFold(n_splits=5)

#Criação do modelo:
modelo = DecisionTreeRegressor()
resultado = cross_val_score(modelo,x,y,cv = kfold, scoring='neg_mean_absolute_error')

#Imprimindo o score:
print ("Erro médio absoluto: ", resultado.mean())

## Bloco7
from sklearn.model_selection import train_test_split
from sklearn.metrics import median_absolute_error

#Separando dados entre treino e teste
X_treino, X_teste, Y_treino, Y_teste = train_test_split(x, y, test_size = 0.3)

#Criação do modelo:
modelo = DecisionTreeRegressor()
modelo.fit(X_treino, Y_treino)

#Imprimindo resultados:
predicoes = modelo.predict(X_teste)
erro = median_absolute_error(Y_teste, predicoes)
print('Erro absoluto mediano: ', erro)

