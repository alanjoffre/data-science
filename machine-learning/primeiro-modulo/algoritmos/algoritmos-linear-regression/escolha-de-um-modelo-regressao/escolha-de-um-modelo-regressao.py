# Machine Learning: Escolha de um modelo de Regressao

# Execute os blocos de códigos no Jupyter Notebook, conforme indicação: 

## Bloco1
import pandas as pd
arquivo = pd.read_csv('D:/Ti/Cursos/Vscode/github/machine-learning/algoritmos/algoritmos-linear-regression/escolha-de-um-modelo-regressao/admission_predict.csv')

## Bloco2
arquivo.head()

## Bloco3
arquivo.shape #Verificar o tamanho do dataset: 400 linhas e 9 colunas

## Bloco4
arquivo.dtypes #Todos os dados já estão do tipo númerico

## Bloco5
faltantes = arquivo.isnull().sum()
print(faltantes)

## Bloco6
arquivo.drop('Serial No.', axis=1, inplace=True) #Excluindo coluna com dados desnecessários

## Bloco7
#Separando as variáveis entre preditores e variável target
y = arquivo['Chance of Admit ']
x = arquivo.drop('Chance of Admit ', axis = 1)

## Bloco8
from sklearn.model_selection import train_test_split
x_treino, x_teste, y_treino, y_teste = train_test_split(x, y, test_size=0.30, random_state=45)

## Bloco9
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
    lasso = Lasso(alpha=1.0, max_iter=1000, tol=0.1)
    elastic = ElasticNet(alpha=1, l1_ratio=0.5, tol=0.2, max_iter=5000)
    reg.fit(x_treino, y_treino)
    ridge.fit(x_treino, y_treino)
    lasso.fit(x_treino, y_treino)
    elastic.fit(x_treino, y_treino)
    resul_reg = reg.score(x_teste, y_teste)
    resul_ridge = ridge.score(x_teste, y_teste)
    resul_lasso = lasso.score(x_teste, y_teste)    
    resul_elastic = elastic.score(x_teste, y_teste)
    dic_regmodels = {'Linear':resul_reg, 'Ridge':resul_ridge, 'Lasso':resul_lasso, 'Elastic':resul_elastic}
    melhor_modelo = max(dic_regmodels, key=dic_regmodels.get)
    print('Regressao Linear: ',resul_reg)
    print('Regressao Ridge:  ',resul_ridge)
    print('Regressao Lasso:  ',resul_lasso)
    print('Regressao Elastic:',resul_elastic)
    print('O modelo', melhor_modelo, 'foi considerado o melhor, com o valor de:', dic_regmodels[melhor_modelo])

## Bloco10
modelosregressao(x_treino, y_treino, x_teste, y_teste)








