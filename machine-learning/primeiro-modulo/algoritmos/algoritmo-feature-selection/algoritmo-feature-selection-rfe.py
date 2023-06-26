# Machine Learning: Feature Selection - RFE

# Documentação: https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.RFE.html

# Execute os blocos de códigos no Jupyter Notebook, conforme indicação: 

## Bloco1
import pandas as pd
from sklearn.feature_selection import RFE
from sklearn.linear_model import Ridge
pd.set_option('display.max_columns', 10)
pd.set_option('display.width', 320)
arquivo = pd.read_csv('D:/Github/data-science/machine-learning/algoritmos/algoritmo-feature-selection/admission_predict.csv')
arquivo.drop('Serial No.', axis=1, inplace=True)
print(arquivo.head())

## Bloco2
#Separando as variaveis entre preditoras e variavel target
y = arquivo['Chance of Admit ']
x = arquivo.drop('Chance of Admit ', axis = 1)

#Definindo o algoritmo de Machine learning que será utilizado:
modelo = Ridge()

#RFE:
rfe = RFE(estimator=modelo, n_features_to_select=5)
fit = rfe.fit(x,y)

#Mostrando os resultados:
print('Número de atribustos:', fit.n_features_)
print('Atributos selecionados:', fit.support_)
print('Ranking dos atributos:', fit.ranking_)

## Bloco3
from sklearn.tree import DecisionTreeRegressor

#Separando as variaveis entre preditoras e variavel target
y = arquivo['Chance of Admit ']
x = arquivo.drop('Chance of Admit ', axis = 1)

#Definindo o algoritmo de Machine learning que será utilizado:
modelo = DecisionTreeRegressor()

#RFE:
rfe = RFE(estimator=modelo, n_features_to_select=5)
fit = rfe.fit(x,y)

#Mostrando os resultados:
print('Número de atribustos:', fit.n_features_)
print('Atributos selecionados:', fit.support_)
print('Ranking dos atributos:', fit.ranking_)

## Bloco4



