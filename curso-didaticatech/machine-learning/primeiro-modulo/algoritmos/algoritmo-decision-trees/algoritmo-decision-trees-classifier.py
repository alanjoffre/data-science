# Machine Learning: Decision Trees Classifier- Exercicio

# Documentação: https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html

#Instale o graphviz: https://graphviz.org/download/
#Instale no ANACONDA NAVIGATOR o pacote: graphviz

# Execute os blocos de códigos no Jupyter Notebook, conforme indicação: 

## Bloco1
from sklearn.datasets import load_iris
import pandas as pd
import warnings
iris = load_iris()
x = pd.DataFrame(iris.data, columns=[iris.feature_names])
y = pd.Series(iris.target)
warnings.filterwarnings('ignore')

## Bloco2
x.head()

## Bloco3
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier

#Definindo os valores que serão testados em DecisionTree:
minimos_split = np.array([2,3,4,5,6,7,8])
maximo_nivel = np.array([3,4,5,6])
algoritmo = ['gini', 'entropy']
valores_grid = {'min_samples_split': minimos_split, 'max_depth': maximo_nivel, 'criterion':algoritmo}

#Criação do modelo:
modelo = DecisionTreeClassifier()

#Criando os grids:
gridDecisionTree = GridSearchCV(estimator = modelo, param_grid = valores_grid, cv=5)
gridDecisionTree.fit(x,y)

#Imprimindo os melhores parâmetros:
print ("Mínimo split: ", gridDecisionTree.best_estimator_.min_samples_split)
print ("Máximo profundidade: ", gridDecisionTree.best_estimator_.max_depth)
print ("Algoritmo escolhido: ", gridDecisionTree.best_estimator_.criterion)
print ("Acurácia: ", gridDecisionTree.best_score_)

## Bloco4
pip install graphviz

## Bloco5
import graphviz
from sklearn.tree import DecisionTreeClassifier, export_graphviz

## Bloco6
#Criando o arquivo que irá armazenar a árvore:
arquivo = 'D:/Github/data-science/curso-didaticatech/machine-learning/primeiro-modulo/algoritmos/algoritmo-decision-trees/exemplo.dot'
melhor_modelo = DecisionTreeClassifier(min_samples_split=2, max_depth=3, criterion='gini')
melhor_modelo.fit(x,y)

#Gerando o gráfico da árvore de decisão:
export_graphviz(melhor_modelo, out_file = arquivo, feature_names = iris.feature_names)
with open(arquivo) as aberto:
    grafico_dot = aberto.read()
h = graphviz.Source(grafico_dot)
h.view()