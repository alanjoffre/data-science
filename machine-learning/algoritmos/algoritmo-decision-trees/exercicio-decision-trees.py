# Machine Learning: Decision Trees - Exercicio

# Execute os blocos de códigos no Jupyter Notebook, conforme indicação: 

## Bloco1
import pandas as pd
arquivo = pd.read_csv('D:/Github/data-science/machine-learning/algoritmos/algoritmo-decision-trees/column_2c_weka.csv')

## Bloco2
arquivo.head()

## Bloco3
faltantes = arquivo.isnull().sum()
faltantes_percentual = (arquivo.isnull().sum() / len(arquivo['pelvic_incidence'])) * 100
print(faltantes_percentual)

## Bloco4
arquivo.dtypes

## Bloco5
#Importante: Todas as variáveis em Decision Trees, precisam ser do tipo: numérico!
# Alterando o nome: Abnormal=1 e Normal=0 no dataset
arquivo['class'] = arquivo['class'].replace('Abnormal', 1)
arquivo['class'] = arquivo['class'].replace('Normal', 0)

## Bloco6
arquivo

## Bloco7
#Separando as variáveis entre preditoras e variável target
y = arquivo['class']
x = arquivo.drop('class', axis = 1)

## Bloco8
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV

#Definindo os valores que serão testados em DecisionTree:
minimos_split = np.array([2, 3, 4, 5, 6, 7, 8])
maximo_nivel = np.array([3, 4, 5, 6])
algoritmo = ['gini', 'entropy']
valores_grid = {'min_samples_split': minimos_split, 'max_depth': maximo_nivel, 'criterion': algoritmo}

#Criação do modelo:
modelo = DecisionTreeClassifier()

#Criando os grids:
gridDecisionTree = GridSearchCV(estimator = modelo, param_grid = valores_grid, cv=5)
gridDecisionTree.fit(x,y)

#Imprimindo os melhores parâmetros:
print ("Mínimo split: ", gridDecisionTree.best_estimator_.min_samples_split)
print ("Máxima profundidade: ", gridDecisionTree.best_estimator_.max_depth)
print ("Algotitmo escolhido: ", gridDecisionTree.best_estimator_.criterion)
print ("Acurácia: ", gridDecisionTree.best_score_)
