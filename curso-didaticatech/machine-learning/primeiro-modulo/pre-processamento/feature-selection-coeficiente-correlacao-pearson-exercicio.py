# Machine Learning: Feature selection - Coeficiente de Correlação Pearson - Exercicio

# Execute os blocos de códigos no Jupyter Notebook, conforme indicação: 

## Bloco1
import pandas as pd
pd.set_option('display.max_columns', 320)
dados = pd.read_csv('D:/Ti/Cursos/Vscode/github/machine-learning/pre-processamento/2015-building-energy-benchmarking.csv')
pd.set_option('display.max_columns', 42)
dados.head()

## Bloco2
import seaborn as sns
import matplotlib.pyplot as plt
plt.figure(figsize=(10,10))
sns.heatmap(dados.corr())
plt.show()




