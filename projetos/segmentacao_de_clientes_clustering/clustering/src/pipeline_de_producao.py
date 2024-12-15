import pandas as pd
import yaml
import logging
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# Carregar configurações do YAML
config_path = r'D:\Github\data-science\projetos\segmentacao_de_clientes_clustering\clustering\config\config.yaml'
with open(config_path, 'r') as file:
    config = yaml.safe_load(file)

# Configurar logging
logging.basicConfig(filename=config['paths']['logs'] + 'app.log', 
                    level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# Diretórios
data_path = config['data']['source']
logs_path = config['paths']['logs']
figures_path = config['paths']['figures']
reports_path = config['paths']['reports']
models_path = config['paths']['models']

# Carregar dados
logging.info("Carregando os dados.")
df = pd.read_csv(data_path)

# Traduzir nomes das colunas
df.columns = [
    'Identificação', 'Ano de Nascimento', 'Escolaridade', 'Estado Civil', 'Renda',
    'Crianças em Casa', 'Adolescentes em Casa', 'Data de Cadastro', 'Recência', 'Gasto em Vinhos',
    'Gasto em Frutas', 'Gasto em Produtos de Carne', 'Gasto em Produtos de Peixe', 'Gasto em Produtos Doces', 
    'Gasto em Produtos de Ouro', 'Compras em Promoções', 'Compras pela Internet', 'Compras por Catálogo',
    'Compras na Loja', 'Visitas ao Site por Mês', 'Aceitou Campanha 3', 'Aceitou Campanha 4', 'Aceitou Campanha 5',
    'Aceitou Campanha 1', 'Aceitou Campanha 2', 'Reclamação', 'Custo de Contato', 'Receita', 'Resposta'
]
logging.info("Nomes das colunas traduzidos.")

# Verificar as primeiras linhas do dataframe
logging.info("Verificando as primeiras linhas do dataframe.")
print(df.head())

# Verificar informações sobre o dataframe
logging.info("Verificando informações do dataframe.")
print(df.info())

# Remover duplicatas
logging.info("Removendo duplicatas.")
df = df.drop_duplicates()

# Verificar valores ausentes
logging.info("Verificando valores ausentes.")
print(df.isnull().sum())

# Imputar valores ausentes na coluna 'Renda' (exemplo com média)
logging.info("Imputando valores ausentes na coluna 'Renda'.")
df['Renda'] = df['Renda'].fillna(df['Renda'].mean())

# Verificar novamente se há valores ausentes
logging.info("Verificando novamente se há valores ausentes.")
print(df.isnull().sum())

# Verificar range de valores para possíveis erros
logging.info("Verificando range de valores para possíveis erros.")
print(df['Ano de Nascimento'].describe())
print(df['Renda'].describe())

# Corrigir dados fora de intervalo razoável
logging.info("Corrigindo dados fora de intervalo razoável.")
df = df[df['Ano de Nascimento'] > config['preprocessing']['remove_outliers']['year_birth']]  # Excluir registros com anos de nascimento não razoáveis
df = df[df['Renda'] < config['preprocessing']['remove_outliers']['income']]  # Excluir valores de renda anômalos

# Selecionar colunas para normalização
logging.info("Normalizando as colunas.")
cols_to_normalize = [
    'Renda', 'Gasto em Vinhos', 'Gasto em Frutas', 'Gasto em Produtos de Carne',
    'Gasto em Produtos de Peixe', 'Gasto em Produtos Doces', 'Gasto em Produtos de Ouro',
    'Compras em Promoções', 'Compras pela Internet', 'Compras por Catálogo',
    'Compras na Loja', 'Visitas ao Site por Mês'
]
scaler = StandardScaler()
df[cols_to_normalize] = scaler.fit_transform(df[cols_to_normalize])

# Verificar dados normalizados
logging.info("Dados normalizados.")
print(df.head())

# Exemplo de histogramas
logging.info("Gerando histograma de renda.")
plt.hist(df['Renda'], bins=30, edgecolor='k')
plt.xlabel('Renda')
plt.ylabel('Frequência')
plt.title('Distribuição de Renda')
plt.savefig(figures_path + 'histograma_renda.png')
plt.show()

# Exemplo de box plots
logging.info("Gerando box plot de renda por nível de escolaridade.")
sns.boxplot(x='Escolaridade', y='Renda', data=df)
plt.title('Renda por Nível de Escolaridade')
plt.savefig(figures_path + 'boxplot_renda_escolaridade.png')
plt.show()

# Método do Cotovelo
logging.info("Calculando Método do Cotovelo.")
sse = []
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, n_init=10, random_state=42)
    kmeans.fit(df[cols_to_normalize])
    sse.append(kmeans.inertia_)
    logging.info(f'K={k}, SSE={kmeans.inertia_}')
    print(f'K={k}, SSE={kmeans.inertia_}')

plt.plot(range(1, 11), sse)
plt.xlabel('Número de Clusters (K)')
plt.ylabel('SSE')
plt.title('Método do Cotovelo')
plt.savefig(figures_path + 'metodo_cotovelo.png')
plt.show()

# Análise do Coeficiente Silhueta
logging.info("Calculando Coeficiente Silhueta.")
silhouette_scores = []
for k in range(2, 11):
    kmeans = KMeans(n_clusters=k, n_init=10, random_state=42)
    kmeans.fit(df[cols_to_normalize])
    score = silhouette_score(df[cols_to_normalize], kmeans.labels_)
    silhouette_scores.append(score)
    logging.info(f'K={k}, Silhouette Score={score}')
    print(f'K={k}, Silhouette Score={score}')

plt.plot(range(2, 11), silhouette_scores)
plt.xlabel('Número de Clusters (K)')
plt.ylabel('Coeficiente Silhueta')
plt.title('Análise do Coeficiente Silhueta')
plt.savefig(figures_path + 'coeficiente_silhueta.png')
plt.show()

# Treinar o modelo KMeans com k=5
logging.info("Treinando modelo KMeans com k=5.")
k_escolhido = config['model']['params']['n_clusters']
kmeans = KMeans(n_clusters=k_escolhido, n_init=config['model']['params']['n_init'], random_state=config['model']['params']['random_state'])
kmeans.fit(df[cols_to_normalize])
labels = kmeans.labels_

# Atribuir cada cliente ao cluster correspondente
logging.info("Atribuindo clusters aos clientes.")
df['Cluster'] = labels

# Recuperar centros dos clusters e inverter a normalização
logging.info("Recuperando centros dos clusters.")
cluster_centers = scaler.inverse_transform(kmeans.cluster_centers_)
clusters = pd.DataFrame(cluster_centers, columns=cols_to_normalize)
clusters['Cluster'] = range(k_escolhido)
print(clusters)

# Salvar modelo
logging.info("Salvando modelo KMeans.")
model_path = models_path + 'kmeans_model.pkl'
joblib.dump(kmeans, model_path)

# Visualizar clusters
logging.info("Gerando visualização dos clusters.")
sns.scatterplot(x='Renda', y='Gasto em Vinhos', hue='Cluster', data=df, palette='viridis')
plt.title('Segmentação de Clientes por Renda e Gasto em Vinhos')
plt.savefig(figures_path + 'scatterplot_renda_vinhos.png')
plt.show()

logging.info("Processo concluído.")
