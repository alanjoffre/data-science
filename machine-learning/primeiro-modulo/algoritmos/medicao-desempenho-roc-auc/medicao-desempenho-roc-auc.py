# Machine Learning: Medição de desempenho - ROC/AUC

# Documentação: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_curve.html

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
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

#Separando os dados entre treino e teste:
X_treino, X_teste, Y_treino, Y_teste = train_test_split(x, y, test_size = 0.3, random_state = 9)

#Criação do modelo:
modelo = LogisticRegression(C=95, penalty='l2') #solver='lbfgs', max_iter=10000
modelo.fit(X_treino, Y_treino)

#Score:
resultado = modelo.score(X_teste, Y_teste)
print('Acurácia:', resultado)

## Bloco8
predicoes = modelo.predict_proba(X_teste)

## Bloco9
predicoes

## Bloco10
probs = predicoes[:, 1]

## Bloco11
from sklearn.metrics import roc_curve
fpr, tpr, thresholds = roc_curve(Y_teste, probs)

## Bloco12
print('TPR:', tpr)
print('FPR:', fpr)
print('Thresholds:', thresholds)

## Bloco13
import matplotlib.pyplot as plt
plt.scatter(fpr, tpr)
plt.show()

## Bloco14
from sklearn.metrics import roc_auc_score
print(roc_auc_score(Y_teste, probs))









