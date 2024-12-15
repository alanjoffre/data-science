import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer
import yaml
import logging
import joblib

# Configurar logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# Carregar configurações do YAML
config_path = r'D:\Github\data-science\projetos\segmentacao_de_clientes_clustering\clustering\config\config.yaml'
with open(config_path, 'r') as file:
    config = yaml.safe_load(file)

# Diretórios
data_path = config['data']['source']

# Configurações do Streamlit
st.set_page_config(page_title="Dashboard de Segmentação de Clientes", layout="wide")

# Título do Dashboard
st.title("Dashboard de Segmentação de Clientes")
st.markdown("Este dashboard apresenta uma análise detalhada dos clusters de clientes para suporte à tomada de decisões.")

# Função para carregar e processar os dados
@st.cache_data
def load_data():
    # Carregar dados do arquivo CSV
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

    # Remover duplicatas
    df = df.drop_duplicates()

    # Imputar valores faltantes na coluna 'Renda' (exemplo com média)
    df['Renda'] = df['Renda'].fillna(df['Renda'].mean())

    # Filtrar dados fora de intervalo razoável
    df = df[df['Ano de Nascimento'] > config['preprocessing']['remove_outliers']['year_birth']]
    df = df[df['Renda'] < config['preprocessing']['remove_outliers']['income']]

    # Calcular Gastos Totais
    df['Gastos Totais'] = df[['Gasto em Vinhos', 'Gasto em Frutas', 'Gasto em Produtos de Carne', 'Gasto em Produtos de Peixe', 'Gasto em Produtos Doces', 'Gasto em Produtos de Ouro']].sum(axis=1)

    # Normalizar os dados
    cols_to_normalize = [
        'Renda', 'Gasto em Vinhos', 'Gasto em Frutas', 'Gasto em Produtos de Carne',
        'Gasto em Produtos de Peixe', 'Gasto em Produtos Doces', 'Gasto em Produtos de Ouro',
        'Compras em Promoções', 'Compras pela Internet', 'Compras por Catálogo',
        'Compras na Loja', 'Visitas ao Site por Mês'
    ]
    scaler = StandardScaler()
    df[cols_to_normalize] = scaler.fit_transform(df[cols_to_normalize])

    # Criar clusters usando KMeans
    k_escolhido = config['model']['params']['n_clusters']
    kmeans = KMeans(n_clusters=k_escolhido, n_init=config['model']['params']['n_init'], random_state=config['model']['params']['random_state'])
    df['Cluster'] = kmeans.fit_predict(df[cols_to_normalize])

    return df, scaler

# Carregar dados
df, scaler = load_data()

# Inverter normalização para exibir gráficos com valores originais
cols_to_normalize = [
    'Renda', 'Gasto em Vinhos', 'Gasto em Frutas', 'Gasto em Produtos de Carne',
    'Gasto em Produtos de Peixe', 'Gasto em Produtos Doces', 'Gasto em Produtos de Ouro',
    'Compras em Promoções', 'Compras pela Internet', 'Compras por Catálogo',
    'Compras na Loja', 'Visitas ao Site por Mês'
]
df_original = df.copy()
df_original[cols_to_normalize] = scaler.inverse_transform(df[cols_to_normalize])

# Menu lateral
menu = st.sidebar.selectbox(
    'Menu',
    ['Análise Descritiva', 'Gráficos Principais', 'Contexto dos Clusters', 'Dados dos Clientes', 'Recomendações']
)

# Análise descritiva dos clusters
if menu == 'Análise Descritiva':
    st.header("Análise Descritiva dos Clusters")
    st.write(df_original.describe())

# Gráficos
if menu == 'Gráficos Principais':
    st.header("Gráficos Principais")

    # Filtro por faixa de renda
    faixa_renda = st.sidebar.slider('Selecione a faixa de renda', min_value=int(df_original['Renda'].min()), max_value=int(df_original['Renda'].max()), value=(int(df_original['Renda'].min()), int(df_original['Renda'].max())))
    df_filtrado = df_original[(df_original['Renda'] >= faixa_renda[0]) & (df_original['Renda'] <= faixa_renda[1])]

    # Layout de colunas para os gráficos
    col1, col2, col3 = st.columns([1, 1, 1])

    sns.set_theme(style="whitegrid")

    with col1:
        st.subheader("Renda vs Gasto em Vinhos")
        fig, ax = plt.subplots(figsize=(5, 4))
        sns.scatterplot(data=df_filtrado, x='Renda', y='Gasto em Vinhos', hue='Cluster', palette='viridis', ax=ax, s=80, edgecolor="w", alpha=0.7)
        ax.set_title('Renda vs Gasto em Vinhos', fontsize=8)
        ax.set_xlabel('Renda', fontsize=6)
        ax.set_ylabel('Gasto em Vinhos', fontsize=6)
        sns.despine()
        st.pyplot(fig)

    with col2:
        st.subheader("Boxplot de Renda por Cluster")
        fig, ax = plt.subplots(figsize=(5, 4))
        sns.boxplot(data=df_filtrado, x='Cluster', y='Renda', palette='viridis', ax=ax)
        ax.set_title('Boxplot de Renda por Cluster', fontsize=8)
        ax.set_xlabel('Cluster', fontsize=6)
        ax.set_ylabel('Renda', fontsize=6)
        sns.despine()
        st.pyplot(fig)

    with col3:
        st.subheader("Histograma de Gastos Totais")
        fig, ax = plt.subplots(figsize=(5, 4))
        sns.histplot(df_filtrado['Gastos Totais'], kde=True, bins=30, color='purple', ax=ax)
        ax.set_title('Distribuição de Gastos Totais', fontsize=8)
        ax.set_xlabel('Gastos Totais', fontsize=6)
        ax.set_ylabel('Frequência', fontsize=6)
        sns.despine()
        st.pyplot(fig)

    # Gráficos Adicionais
    col4, col5, col6 = st.columns([1, 1, 1])

    with col4:
        st.subheader("Gráfico de Barras: Compras por Categoria")
        fig, ax = plt.subplots(figsize=(5, 4))
        sns.barplot(data=df_filtrado.melt(id_vars='Cluster', value_vars=['Compras em Promoções', 'Compras pela Internet', 'Compras por Catálogo', 'Compras na Loja']), x='variable', y='value', hue='Cluster', palette='viridis', ax=ax)
        ax.set_title('Compras por Categoria', fontsize=8)
        ax.set_xlabel('Categoria de Compra', fontsize=6)
        ax.set_ylabel('Quantidade', fontsize=6)
        plt.xticks(rotation=45, fontsize=6)
        sns.despine()
        st.pyplot(fig)

    with col5:
        st.subheader("Gráfico de Linhas: Visitas ao Site por Mês")
        fig, ax = plt.subplots(figsize=(5, 4))
        sns.lineplot(data=df_filtrado, x='Cluster', y='Visitas ao Site por Mês', hue='Cluster', palette='viridis', marker='o', ax=ax)
        ax.set_title('Visitas ao Site por Mês', fontsize=8)
        ax.set_xlabel('Cluster', fontsize=6)
        ax.set_ylabel('Visitas ao Site', fontsize=6)
        sns.despine()
        st.pyplot(fig)

    with col6:
        st.subheader("Gráfico de Dispersão: Renda vs. Compras pela Internet")
        fig, ax = plt.subplots(figsize=(5, 4))
        sns.scatterplot(data=df_filtrado, x='Renda', y='Compras pela Internet', hue='Cluster', palette='viridis', ax=ax, s=80, edgecolor="w", alpha=0.7)
        ax.set_title('Renda vs. Compras pela Internet', fontsize=8)
        ax.set_xlabel('Renda', fontsize=6)
        ax.set_ylabel('Compras pela Internet', fontsize=6)
        sns.despine()
        st.pyplot(fig)

# Contexto dos Clusters
if menu == 'Contexto dos Clusters':
    st.header("Contexto dos Clusters")
    cluster_context = {
        0: "Clientes com renda média e altos gastos em vinhos e carne. Provavelmente estão interessados em produtos premium de vinhos e carnes.",
        1: "Clientes com renda baixa e gastos baixos em todas as categorias. Eles podem ser mais sensíveis a preços e interessados em promoções e ofertas.",
        2: "Clientes com alta renda e altos gastos em todas as categorias, especialmente vinhos e carne. Provavelmente valorizam produtos de alta qualidade e experiências premium.",
        3: "Clientes com renda alta e altos gastos em vinhos, carne e frutas. Eles podem estar interessados em produtos saudáveis e de alta qualidade.",
        4: "Clientes com renda média-baixa e gastos moderados em vinhos e carne. Eles podem apreciar ofertas e produtos com bom custo-benefício."
    }
    for cluster, context in cluster_context.items():
        st.markdown(f"<p style='font-size:12px'><strong>Cluster {cluster}:</strong> {context}</p>", unsafe_allow_html=True)

# Mostrar dados
if menu == 'Dados dos Clientes':
    st.header("Dados dos Clientes")
    st.dataframe(df_original)

# Recomendações
if menu == 'Recomendações':
    st.header("Recomendações")
    st.markdown("""
    - **Cluster 0**: Foque em promoções de vinhos e carne premium.
    - **Cluster 1**: Ofereça descontos e promoções para atrair clientes sensíveis a preços.
    - **Cluster 2**: Invista em produtos de alta qualidade e experiências exclusivas.
    - **Cluster 3**: Promova produtos saudáveis e premium.
    - **Cluster 4**: Destaque ofertas com bom custo-benefício.
    """)
