# Machine Learning: Regressão Linear
# Documentação: https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html

# Execute os blocos de códigos no Jupyter Notebook, conforme indicação: 

## Bloco1
from sklearn.datasets import make_regression
import matplotlib.pyplot as plt
# gerando uma massa de dados:
x, y = make_regression(n_samples=200, n_features=1, noise=30)
# mostrando no gráfico
plt.scatter(x,y)
plt.show()

## Bloco2
from sklearn.linear_model import LinearRegression
# Criação do modelo:
modelo = LinearRegression()

## Bloco3
modelo.fit(x,y)

# Bloco4
modelo.intercept_ #coeficiente linear
# resultado: -2.1464654545591593

## Bloco5
modelo.coef_ #coeficiente angular
# resultado: array([66.49840473])

## Bloco6

# mostrando o resultado
import numpy as np
plt.scatter(x,y)
xreg = np.arange(-3, 3, 1)
plt.plot(xreg, 66.49*xreg-2.146, color='red') # gráfico regressão
plt.show()

