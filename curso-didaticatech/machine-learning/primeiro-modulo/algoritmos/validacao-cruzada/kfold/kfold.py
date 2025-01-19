# Machine Learning: Validação Cruzada - KFold

# Execute os blocos de códigos no Jupyter Notebook, conforme indicação: 

## Bloco1
import pandas as pd
arquivo = pd.read_csv('D:/Github/machine-learning/algoritmos/validacao-cruzada/kfold/Admission_Predict.csv')

## Bloco2
arquivo.head()

## Bloco3
arquivo.drop('Serial No.', axis=1, inplace=True)

## Bloco4
#Separando as variáveis entre preditoras e variável target
y = arquivo['Chance of Admit ']
x = arquivo.drop('Chance of Admit ', axis=1)

## Bloco5
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold

## Bloco6
def modelosregressaokfold(a, b):
    from sklearn.linear_model import LinearRegression
    from sklearn.linear_model import Ridge
    from sklearn.linear_model import Lasso
    from sklearn.linear_model import ElasticNet
    kfold = KFold(n_splits=10, shuffle=True)
    x = a
    y = b
    reg = LinearRegression()
    ridge = Ridge()
    lasso = Lasso()
    elastic = ElasticNet()
    resul_reg = cross_val_score(reg,x,y,cv = kfold)
    resul_ridge = cross_val_score(ridge,x,y,cv = kfold)
    resul_lasso = cross_val_score(lasso,x,y,cv = kfold)
    resul_elastic = cross_val_score(elastic,x,y,cv = kfold)
    dic_regmodels = {'Linear':resul_reg.mean(), 'Ridge':resul_ridge.mean(), 'Lasso':resul_lasso.mean(), 'Elastic':resul_elastic.mean()}
    melhor_modelo = max(dic_regmodels, key=dic_regmodels.get)
    print('Regressao Linear: ',resul_reg)
    print('Regressao Ridge:  ',resul_ridge)
    print('Regressao Lasso:  ',resul_lasso)
    print('Regressao Elastic:',resul_elastic)
    print('O modelo', melhor_modelo, 'foi considerado o melhor, com o valor de:', dic_regmodels[melhor_modelo])

## Bloco7
modelosregressaokfold(x,y)










