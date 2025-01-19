# Machine Learning: Logistic Regression - Ajuste Fino de Parâmetros

# Execute os blocos de códigos no Jupyter Notebook, conforme indicação: 

## Bloco1
import pandas as pd
pd.set_option('display.max_columns', 64)
pd.set_option('display.max_rows', 64)
arquivo = pd.read_csv('D:/Github/machine-learning/algoritmos/algoritmo-logistic-regression/logistic-regression-ajuste-fino-de-parametros/data_train_reduced.csv')

## Bloco2
arquivo.head()

## Bloco3
arquivo.shape

## Bloco4
arquivo.dtypes

## Bloco5
faltantes = arquivo.isnull().sum()
faltantes_percentual = (arquivo.isnull().sum() / len(arquivo['Product'])) * 100
print(faltantes_percentual)

## Bloco6
arquivo.drop('q8.20', axis=1, inplace=True)
arquivo.drop('q8.18', axis=1, inplace=True)
arquivo.drop('q8.17', axis=1, inplace=True)
arquivo.drop('q8.8', axis=1, inplace=True)
arquivo.drop('q8.9', axis=1, inplace=True)
arquivo.drop('q8.10', axis=1, inplace=True)
arquivo.drop('q8.2', axis=1, inplace=True)
arquivo.drop('Respondent.ID', axis=1, inplace=True)
arquivo.drop('Product', axis=1, inplace=True)
arquivo.drop('q1_1.personal.opinion.of.this.Deodorant', axis=1, inplace=True)#Esta coluna proporciona acurácia de 100%(nota)

## Bloco7
# > 20% = Exclua a coluna
# >= 20%= Altere os dados faltantes pelo cáculo da mediana
arquivo['q8.12'].fillna(arquivo['q8.12'].median(), inplace=True)
arquivo['q8.7'].fillna(arquivo['q8.12'].median(), inplace=True)

## Bloco8
faltantes = arquivo.isnull().sum()
faltantes_percentual = (arquivo.isnull().sum() / len(arquivo['Product.ID'])) * 100
print(faltantes_percentual)

## Bloco9
#Separando as variáveis entre preditoras e variável target
y = arquivo['Instant.Liking']
x = arquivo.drop('Instant.Liking', axis = 1)

## Bloco10
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













