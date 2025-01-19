# Machine Learning: Feature selection - Correlação

# Execute os blocos de códigos no Jupyter Notebook, conforme indicação: 

## Bloco1
import pandas as pd
pd.set_option('display.width', 320) #Definir tamanho da tela
colunas = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
dados = pd.read_csv('D:/Ti/Cursos/Vscode/github/machine-learning/pre-processamento/pima-indians-diabetes.csv', names = colunas)
print(dados.corr(method = 'pearson'))#Função corr / método: pearson

## Bloco2
import pandas as pd
import seaborn as sns #biblioteca seaborn / Utilizada: Dados e Estatística
import matplotlib.pyplot as plt
colunas = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
dados = pd.read_csv('D:/Ti/Cursos/Vscode/github/machine-learning/pre-processamento/pima-indians-diabetes.csv', names = colunas)
plt.figure(figsize=(5,5)) #Tamanho da figura
sns.heatmap(dados.corr()) #Heatmap: Chama a função (mapa de calor)



