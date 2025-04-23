# Machine Learning: Ridge Regression

# Para encontrar a documentação de uma função: sklearn ridge
# Documentação: https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html

# Execute os blocos de códigos no Jupyter Notebook, conforme indicação: 

## Bloco1
import pandas as pd
pd.set_option('display.max_columns', 21)
arquivo = pd.read_csv('D:/Ti/Cursos/Vscode/github/machine-learning/algoritmos/algoritmos-linear-regression/ridge-regression/kc_house_data.csv')

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
resultadoRidge = modeloRidge.score(x_teste, y_teste) # Ridge Regression
print(resultadoRidge)






