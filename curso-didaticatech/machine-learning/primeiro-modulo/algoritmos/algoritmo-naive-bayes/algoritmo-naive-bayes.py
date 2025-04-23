# Machine Learning: Naive Bayes - Exercicio

# Execute os blocos de códigos no Jupyter Notebook, conforme indicação: 

## Bloco1
#Carregando o conjunto de dados
import pandas as pd
arquivo = pd.read_csv('D:/Github/data-science/machine-learning/algoritmos/algoritmo-naives-bayes/wine_dataset.csv')

## Bloco2
arquivo.head()

## Bloco3
#Separando as variáveis entre preditoras e variável alvo
y = arquivo['style']
x = arquivo.drop('style', axis = 1)

#Separando os dados entre treino e teste:
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
#Criação do modelo:
modelo = GaussianNB()
skfold = StratifiedKFold(n_splits=3)
resultado = cross_val_score(modelo,x,y,cv = skfold)
print(resultado.mean())
