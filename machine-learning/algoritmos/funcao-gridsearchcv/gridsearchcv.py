# Machine Learning: Função - GridSearchCV

# Execute os blocos de códigos no Jupyter Notebook, conforme indicação: 

## Bloco1
import pandas as pd
arquivo = pd.read_csv('D:/Github/data-science/machine-learning/algoritmos/funcao-gridsearchcv/Admission_Predict.csv')
arquivo.drop('Serial No.', axis=1, inplace=True)

## Bloco2
arquivo.head()

## Bloco3
#Separando as variáveis entre preditoras e variável target
y = arquivo['Chance of Admit ']
x = arquivo.drop('Chance of Admit ', axis=1)

## Bloco4
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import ElasticNet

## Bloco5
#Definindo os valores que serão testados:
valores = {'alpha': [0.1,0.5,1,2,5,10,25,50,100], 'l1_ratio': [0.02,0.05,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8]}

## Bloco6
#Criando o modelo:
modelo = ElasticNet()
procura = GridSearchCV(estimator = modelo, param_grid=valores, cv=5)# Importante: Fará todas as combinações possiveis.
procura.fit(x,y)

#Imprimindo o resultado:
print ('Melhor score:', procura.best_score_)
print ('Melhor alpha:', procura.best_estimator_.alpha)
print ('Melhor l1_ratio:', procura.best_estimator_.l1_ratio)








