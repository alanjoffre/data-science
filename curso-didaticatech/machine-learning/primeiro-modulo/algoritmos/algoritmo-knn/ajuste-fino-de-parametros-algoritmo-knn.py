# Machine Learning: KNN - Ajuste fino de parâmetros

# Documentação: https://scikit-learn.org/0.24/modules/generated/sklearn.neighbors.DistanceMetric.html
#               https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html

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
print(x.shape, y.shape)

## Bloco6
#Verificar se há o balanceamento das variáveis
y.value_counts()

## Bloco7
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler #IMPORTANTE: KNN precisa de normalização dos dados
from sklearn.model_selection import GridSearchCV

#Normalizando as variáveis preditoras:
normalizador = MinMaxScaler(feature_range = (0,1))
X_norm = normalizador.fit_transform(x)

#Definindo os valores que serão testados no KNN:
valores_K = np.array([3,5,7,9,11])
calculo_distancia = ['minkowski', 'chebyshev']
valores_p = np.array([1,2,3,4])
valores_grid = {'n_neighbors':valores_K, 'metric':calculo_distancia, 'p':valores_p}

#Criação do modelo:
modelo = KNeighborsClassifier()

#Criando os grids:
gridKNN = GridSearchCV(estimator = modelo, param_grid = valores_grid, cv=5)
gridKNN.fit(X_norm, y)

#Imprimindo os melhores parâmetros:
print ('Melhor acurácia: ', gridKNN.best_score_)
print ('Melhor K: ', gridKNN.best_estimator_.n_neighbors)
print ('Melhor distância: ', gridKNN.best_estimator_.metric)
print ('Melhor valor p: ', gridKNN.best_estimator_.p)








