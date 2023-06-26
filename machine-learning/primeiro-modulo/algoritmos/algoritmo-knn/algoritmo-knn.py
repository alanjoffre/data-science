# Machine Learning: KNN

# Documentação: https://scikit-learn.org/0.24/modules/generated/sklearn.neighbors.DistanceMetric.html
#               https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html

# Execute os blocos de códigos no Jupyter Notebook, conforme indicação: 

## Bloco1
from sklearn.datasets import load_breast_cancer
import pandas as pd
import warnings
pd.set_option('display.max_columns', 30)
warnings.filterwarnings('ignore')
dados = load_breast_cancer()
x = pd.DataFrame(dados.data, columns=[dados.feature_names])
y = pd.Series(dados.target)

## Bloco2
print(dados)

## Bloco3
type(dados)

## Bloco4
x.head()

## Bloco5
print(x.shape, y.shape)

## Bloco6
#Verificar se há o balanceamento das variáveis
y.value_counts()

## Bloco7
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler #IMPORTANTE: KNN precisa de normalização dos dados
from sklearn.model_selection import train_test_split

#Normalizando as variáveis preditoras:
normalizador = MinMaxScaler(feature_range = (0,1))
X_norm = normalizador.fit_transform(x)

#Separando os dados entre treino e teste:
X_treino, X_teste, Y_treino, Y_teste = train_test_split(X_norm, y, test_size = 0.3, random_state = 16)

#Criação do modelo
modelo = KNeighborsClassifier(n_neighbors=5) #Default = 5
modelo.fit(X_treino, Y_treino)

#Score:
resultado = modelo.score(X_teste, Y_teste)
print("Acurácia:", resultado)

## Exemplo sem normalizar
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

#Separando os dados entre treino e teste:
X_treino, X_teste, Y_treino, Y_teste = train_test_split(x, y, test_size = 0.3, random_state = 16)

#Criação do modelo
modelo = KNeighborsClassifier(n_neighbors=5) #Default = 5
modelo.fit(X_treino, Y_treino)

#Score:
resultado = modelo.score(X_teste, Y_teste)
print("Acurácia:", resultado)








