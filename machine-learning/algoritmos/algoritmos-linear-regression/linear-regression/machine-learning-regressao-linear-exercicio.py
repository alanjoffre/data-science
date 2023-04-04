# Machine Learning: Regressão Linear - Exercicio

# importe a bibliotecas Pandas: pip install pandas

# Link do dataset: https://www.kaggle.com/harlfoxem/housesalesprediction
# Coloque o arquivo dataset no diretório do projeto.

# D:\Ti\Cursos\Vscode\github\machine-learning\algoritmos\regressao-linear
# Altere as barras
# D:/Ti/Cursos/Vscode/github/machine-learning/algoritmos/regressao-linear

# Execute os blocos de códigos no Jupyter Notebook, conforme indicação: 

## Bloco1
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
pd.set_option('display.max_columns', 21)
arquivo = pd.read_csv('D:/Ti/Cursos/Vscode/github/machine-learning/algoritmos/algoritmos-linear-regression/linear-regression/kc_house_data.csv')

## Bloco2
arquivo.head()

## Bloco3
# Excluindo features irrelevantes
arquivo.drop('id', axis = 1, inplace = True)
arquivo.drop('date', axis = 1, inplace = True)
arquivo.drop('zipcode', axis = 1, inplace = True)
arquivo.drop('lat', axis = 1, inplace = True)
arquivo.drop('long', axis = 1, inplace = True)

## Bloco4
arquivo.head()

## Bloco5
# Definindo variáveis preditoras e variável target
y = arquivo['price']
x = arquivo.drop('price', axis=1)

## Bloco6
# Separando os dados em treino e teste
#random_state = para dar o mesmo resultado em outras máquinas
x_treino, x_teste, y_treino, y_teste = train_test_split(x, y, test_size = 0.3, random_state=10) 

## Bloco7
# Criando o modelo
modelo = LinearRegression()
modelo.fit(x_treino, y_treino)

## Bloco8
# Calculando o coeficiente R2
resultado = modelo.score(x_teste, y_teste) # Linear Regression 
print(resultado)
