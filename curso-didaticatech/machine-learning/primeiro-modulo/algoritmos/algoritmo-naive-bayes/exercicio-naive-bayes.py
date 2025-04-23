# Machine Learning: Naive Bayes - Exercicio

# Documentação: https://scikit-learn.org/stable/modules/naive_bayes.html#gaussian-naive-bayes

# Execute os blocos de códigos no Jupyter Notebook, conforme indicação: 

## Bloco1
from sklearn.datasets import load_iris
import pandas as pd
iris = load_iris()
x = pd.DataFrame(iris.data, columns=[iris.feature_names])
y = pd.Series(iris.target)

## Bloco2
x.head()

## Bloco3
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split

#Separando os dados entre treino e teste:
X_treino, X_teste, Y_treino, Y_teste = train_test_split(x, y, test_size = 0.3, random_state = 67)

#Criação do modelo:
modelo = GaussianNB()
modelo.fit(X_treino, Y_treino)

#Score
resultado = modelo.score(X_teste, Y_teste)
print("Accurácia:", resultado)








