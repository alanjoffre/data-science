# Machine Learning: Pré-processamento
# Normalizando dados em python (utilizando funções, Normalizer, MinMaxScaler, StandarScaler, MaxAbsScaler)

# Execute os blocos de códigos no Jupyter Notebook, conforme indicação: 

## Bloco1
from sklearn.preprocessing import MinMaxScaler 
X = [[4, 1, 2, 2],[1, 3, 9, 3],[5, 7, 5, 1]] 
normalizador = MinMaxScaler(feature_range = (0 , 1)) 
print(normalizador.fit_transform(X))

## Bloco2
from sklearn.preprocessing import StandardScaler 
X = [[4, 1, 2, 2],[1, 3, 9, 3],[5, 7, 5, 1]] 
normalizador = StandardScaler() 
print(normalizador.fit_transform(X))

## Bloco3
from sklearn.preprocessing import MaxAbsScaler 
X = [[4, 1, 2, 2],[1, 3, 9, 3],[5, 7, 5, 1]] 
normalizador = MaxAbsScaler() 
print(normalizador.fit_transform(X))

## Bloco4
from sklearn.preprocessing import normalize 
X = [[4, 1, 2, 2],[1, 3, 9, 3],[5, 7, 5, 1]] 
normalizador = normalize(X,  norm = 'l1') 
print(normalizador)

## Bloco5
from sklearn.preprocessing import normalize 
X = [[4, 1, 2, 2],[1, 3, 9, 3],[5, 7, 5, 1]] 
normalizador = normalize(X, norm = 'l2') 
print(normalizador)

## Bloco6
from sklearn.preprocessing import normalize 
X = [[4, 1, 2, 2],[1, 3, 9, 3],[5, 7, 5, 1]] 
normalizador = normalize(X, norm = 'max', axis=0)# Normalizar por coluna=0 e por linha=1 
print(normalizador)









