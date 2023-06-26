# Machine Learning: Regressão Linear - Coeficiente de Determinação R2
# Documentação: https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html

# Execute os blocos de códigos no Jupyter Notebook, conforme indicação: 

## Bloco1
from sklearn.datasets import make_regression
import matplotlib.pyplot as plt
# gerando uma massa de dados:
x, y = make_regression(n_samples=200, n_features=1, noise=5)
# mostrando no gráfico
plt.scatter(x,y)
plt.show()

## Bloco2
from sklearn.linear_model import LinearRegression
# Criação do modelo:
modelo = LinearRegression()

## Bloco3
modelo.fit(x,y)

## Bloco4
modelo.intercept_ #coeficiente linear

## Bloco5
modelo.coef_ #coeficiente angular

## Bloco6
import numpy as np
plt.scatter(x,y)
xreg = np.arange(-3, 3, 1)
plt.plot(xreg, 78.78*xreg-1.812, color='red') # gráfico regressão
plt.show()

## Bloco7
from sklearn.model_selection import train_test_split
x_treino, x_teste, y_treino, y_teste = train_test_split(x, y, test_size = 30)

## Bloco8
modelo.fit(x_treino, y_treino)

## Bloco9
resultado = modelo.score(x_teste, y_teste) #Linear Regression

## Bloco10
print(resultado)

## Bloco11
import numpy as np
plt.scatter(x_treino,y_treino)
xreg = np.arange(-3, 3, 1)
plt.plot(xreg, 77.745*xreg-0.663, color='red') # gráfico regressão
plt.show()

## Bloco12
import numpy as np
plt.scatter(x_teste,y_teste)
xreg = np.arange(-3, 3, 1)
plt.plot(xreg, 77.745*xreg-0.663, color='red') # gráfico regressão
plt.show()




