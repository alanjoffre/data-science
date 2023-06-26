# Machine Learning: Função - RandomizedSearchCV

# Execute os blocos de códigos no Jupyter Notebook, conforme indicação: 

## Bloco1
import pandas as pd
arquivo = pd.read_csv('D:/Github/machine-learning/algoritmos/funcao-randomizedsearchcv/Admission_Predict.csv')
arquivo.drop('Serial No.', axis=1, inplace=True)

## Bloco2
arquivo.head()

## Bloco3
#Separando as variáveis entre preditoras e variável target
y = arquivo['Chance of Admit ']
x = arquivo.drop('Chance of Admit ', axis=1)

## Bloco4
from sklearn.model_selection import RandomizedSearchCV
from sklearn.linear_model import ElasticNet

## Bloco5
#Definindo os valores que serão testados:
valores = {'alpha': [0.1,0.5,1,2,5,10,25,50,100,150,200,300,500,750,1000,1500,2000,3000,5000], 'l1_ratio': [0.02,0.05,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]}

## Bloco6
#Criando o modelo:
modelo = ElasticNet()
procura = RandomizedSearchCV(estimator = modelo, param_distributions=valores, n_iter=150, cv=5, random_state=15)# Fará somente as combinações citadas.
procura.fit(x,y)

#Imprimindo o resultado:
print ('Melhor score:', procura.best_score_)
print ('Melhor alpha:', procura.best_estimator_.alpha)
print ('Melhor l1_ratio:', procura.best_estimator_.l1_ratio)







