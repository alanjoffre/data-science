# Machine Learning: Elastic Net

# Para encontrar a documentação de uma função: sklearn elastic net
# https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.ElasticNet.html

# Execute os blocos de códigos no Jupyter Notebook, conforme indicação: 

## Bloco1
import pandas as pd
pd.set_option('display.max_columns', 21)
arquivo = pd.read_csv('D:/Ti/Cursos/Vscode/github/machine-learning/algoritmos/algoritmos-linear-regression/elastic-net/kc_house_data.csv')

## Bloco2
arquivo.head()

## Bloco3
arquivo.drop('id', axis = 1, inplace = True)
arquivo.drop('date', axis = 1, inplace = True)
arquivo.drop('zipcode', axis = 1, inplace = True)
arquivo.drop('lat', axis = 1, inplace = True)
arquivo.drop('long', axis = 1, inplace = True)

## Bloco4
arquivo.head()

## Bloco5
from sklearn.linear_model import LinearRegression
modelo = LinearRegression()

## Bloco6
from sklearn.model_selection import train_test_split
x_treino, x_teste, y_treino, y_teste = train_test_split(x, y, test_size=0.30, random_state=14)

## Bloco7
modelo.fit(x_treino, y_treino)
resultado = modelo.score(x_teste, y_teste) # Linear Regression
print(resultado)

## Bloco8
from sklearn.linear_model import Ridge
modeloRidge = Ridge(alpha=1.0) #Alpha por default = 1.0
modeloRidge.fit(x_treino, y_treino)
resultadoRidge = modeloRidge.score(x_teste, y_teste) # Ridge Regression (L1)
print(resultadoRidge)

## Bloco9
from sklearn.linear_model import Lasso
modeloLasso = Lasso(alpha=1000, max_iter=1000, tol=0.1)
modeloLasso.fit(x_treino, y_treino)
resultadoLasso = modeloLasso.score(x_teste,y_teste) #Lasso Regression (L2)
print(resultadoLasso)

## Bloco10
from sklearn.linear_model import ElasticNet
modeloElasticNet = ElasticNet(alpha=1, l1_ratio=0.9, tol=0.2, max_iter=5000) #Default l1_ratio=0.5 (50% L1 e 50% L2)
modeloElasticNet.fit(x_treino, y_treino)
resultadoElasticNet = modeloElasticNet.score(x_teste, y_teste) #ElasticNet
print(resultadoElasticNet)






