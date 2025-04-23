# Machine Learning: Comparando diferentes modelos em um unico código

# Para encontrar a documentação de uma função: 


# Execute os blocos de códigos no Jupyter Notebook, conforme indicação: 

## Bloco1
import pandas as pd
pd.set_option('display.max_columns', 21)
arquivo = pd.read_csv('D:/Ti/Cursos/Vscode/github/machine-learning/algoritmos/algoritmos-linear-regression/comparando-diferentes-modelos-no-codigo/kc_house_data.csv')

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
y = arquivo['price']
x = arquivo.drop('price', axis = 1)

## Bloco6
from sklearn.model_selection import train_test_split
x_treino, x_teste, y_treino, y_teste = train_test_split(x, y, test_size=0.30, random_state=14)

## Bloco7
def modelosregressao(a, b, c, d):
    from sklearn.linear_model import LinearRegression
    from sklearn.linear_model import Ridge
    from sklearn.linear_model import Lasso
    from sklearn.linear_model import ElasticNet
    x_treino = a
    y_treino = b
    x_teste = c
    y_teste = d
    reg = LinearRegression()
    ridge = Ridge(alpha=1.0)
    lasso = Lasso(alpha=1000, max_iter=1000, tol=0.1)
    elastic = ElasticNet(alpha=1, l1_ratio=0.9, tol=0.2, max_iter=5000)
    reg.fit(x_treino, y_treino)
    ridge.fit(x_treino, y_treino)
    lasso.fit(x_treino, y_treino)
    elastic.fit(x_treino, y_treino)
    resul_reg = reg.score(x_teste, y_teste)
    resul_ridge = ridge.score(x_teste, y_teste)
    resul_lasso = lasso.score(x_teste, y_teste)    
    resul_elastic = elastic.score(x_teste, y_teste)
    print('Regressao Linear: ',resul_reg)
    print('Regressao Ridge:  ',resul_ridge)
    print('Regressao Lasso:  ',resul_lasso)
    print('Regressao Elastic:',resul_elastic)

## Bloco8
modelosregressao(x_treino, y_treino, x_teste, y_teste)








