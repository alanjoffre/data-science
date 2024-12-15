import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO

# Carregar dados
@st.cache_data
def carregar_dados():
    return pd.read_csv('D:/Github/data-science/projetos/rotatividade-de-clientes/machine-learning/data/processed/rclientes_dados_tratados.csv')

# Função para salvar gráficos
def salvar_grafico(fig):
    buf = BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)
    return buf

# Página principal do dashboard
def main():
    # Título da página com fundo destacado e fonte menor
    st.markdown(""" 
    <style>
        .dashboard-title {
            background-color: #ff4b4b;
            color: white;
            padding: 10px;
            border-radius: 5px;
            text-align: center;
            font-size: 24px;
        }
        .extra-small-title {
            font-size: 14px;
            font-weight: bold;
            margin-top: 0px;
            margin-bottom: 8px;
        }
        .compact-button {
            padding: 1px 8px;
            font-size: 10px;
        }
    </style>
    <div class="dashboard-title">Dashboard - Análise de Churn de Clientes</div>
    """, unsafe_allow_html=True)

    st.write("Bem-vindo ao Dashboard! Explore visualmente os dados sobre a rotatividade de clientes e obtenha insights valiosos.")

    # Carregar dados
    df = carregar_dados()

    # Armazenar os gráficos gerados
    generated_graphs = []

    # Layout dos gráficos
    st.subheader("Visualizações Iniciais")
    
    # Primeira linha de gráficos
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown('<p class="extra-small-title">Distribuição de CreditScore</p>', unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(4, 3))
        sns.histplot(df['CreditScore'], kde=True, ax=ax)
        ax.set_title("")
        st.pyplot(fig)
        st.download_button("Baixar", data=salvar_grafico(fig), file_name="distribuicao_credit_score.png", mime="image/png", key="download1", help="Baixar gráfico", use_container_width=True)
        generated_graphs.append("CreditScore")

    with col2:
        st.markdown('<p class="extra-small-title">Distribuição por Região</p>', unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(4, 3))
        sns.countplot(x='Geography', data=df, palette="viridis", ax=ax)
        ax.set_title("")
        st.pyplot(fig)
        st.download_button("Baixar", data=salvar_grafico(fig), file_name="distribuicao_geografia.png", mime="image/png", key="download2", help="Baixar gráfico", use_container_width=True)
        generated_graphs.append("Geography")

    with col3:
        st.markdown('<p class="extra-small-title">Distribuição por Gênero</p>', unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(4, 3))
        sns.countplot(x='Gender', data=df, palette="viridis", ax=ax)
        ax.set_title("")
        st.pyplot(fig)
        st.download_button("Baixar", data=salvar_grafico(fig), file_name="distribuicao_genero.png", mime="image/png", key="download3", help="Baixar gráfico", use_container_width=True)
        generated_graphs.append("Gender")

    # Segunda linha de gráficos
    col4, col5, col6 = st.columns(3)
    with col4:
        st.markdown('<p class="extra-small-title">Distribuição Etária</p>', unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(4, 3))
        sns.histplot(df['Age'], bins=20, kde=True, ax=ax)
        ax.set_title("")
        st.pyplot(fig)
        st.download_button("Baixar", data=salvar_grafico(fig), file_name="distribuicao_idade.png", mime="image/png", key="download4", help="Baixar gráfico", use_container_width=True)
        generated_graphs.append("Age")

    with col5:
        st.markdown('<p class="extra-small-title">Clientes com/sem Cartão de Crédito</p>', unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(4, 3))
        sns.countplot(x='HasCrCard', data=df, palette="viridis", ax=ax)
        ax.set_title("")
        st.pyplot(fig)
        st.download_button("Baixar", data=salvar_grafico(fig), file_name="clientes_cartao_credito.png", mime="image/png", key="download5", help="Baixar gráfico", use_container_width=True)
        generated_graphs.append("HasCrCard")

    with col6:
        st.markdown('<p class="extra-small-title">Análise de Churn (Exited)</p>', unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(4, 3))
        sns.countplot(x='Exited', data=df, palette="viridis", ax=ax)
        ax.set_title("")
        st.pyplot(fig)
        st.download_button("Baixar", data=salvar_grafico(fig), file_name="analise_churn.png", mime="image/png", key="download6", help="Baixar gráfico", use_container_width=True)
        generated_graphs.append("Exited")

    # Customização de gráficos
    st.subheader("Criação de Gráficos Customizados")
    st.write("Selecione as variáveis para montar um gráfico personalizado.")

    # Opções para criar gráfico customizado
    colunas_numericas = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
    colunas_categoricas = df.select_dtypes(include=['object']).columns.tolist()

    tipo_grafico = st.selectbox("Escolha o tipo de gráfico", ["Histograma", "Gráfico de Barras", "Gráfico de Linhas", "Gráfico de Densidade", "Box Plot"], key="tipo_grafico_1")

    # Remover gráficos já gerados da seleção
    tipo_grafico_opcoes = ["Histograma", "Gráfico de Barras", "Gráfico de Linhas", "Gráfico de Densidade", "Box Plot"]
    tipo_grafico_opcoes = [opcao for opcao in tipo_grafico_opcoes if opcao not in generated_graphs]

    tipo_grafico_opcoes = st.selectbox("Escolha o tipo de gráfico para a segunda seleção", tipo_grafico_opcoes, key="tipo_grafico_2")

    if tipo_grafico == "Gráfico de Barras":
        coluna_escolhida = st.selectbox("Escolha uma coluna categórica", [col for col in colunas_categoricas if col not in generated_graphs], key="coluna_categorica")
    else:
        coluna_escolhida = st.selectbox("Escolha uma coluna numérica", [col for col in colunas_numericas if col not in generated_graphs], key="coluna_numerica")

    if st.button("Gerar Gráfico"):
        if tipo_grafico == "Histograma" and coluna_escolhida in colunas_numericas:
            fig, ax = plt.subplots(figsize=(6, 4))
            sns.histplot(df[coluna_escolhida], kde=True, ax=ax)
            ax.set_title("")
            st.pyplot(fig)
            st.download_button("Baixar gráfico customizado", data=salvar_grafico(fig), file_name=f"grafico_customizado_{coluna_escolhida}.png", mime="image/png", key="custom_download1")
            generated_graphs.append(coluna_escolhida)

        elif tipo_grafico == "Gráfico de Barras" and coluna_escolhida in colunas_categoricas:
            fig, ax = plt.subplots(figsize=(6, 4))
            sns.countplot(x=coluna_escolhida, data=df, palette="viridis", ax=ax)
            ax.set_title("")
            st.pyplot(fig)
            st.download_button("Baixar gráfico customizado", data=salvar_grafico(fig), file_name=f"grafico_customizado_{coluna_escolhida}.png", mime="image/png", key="custom_download2")
            generated_graphs.append(coluna_escolhida)

        elif tipo_grafico == "Gráfico de Linhas" and coluna_escolhida in colunas_numericas:
            fig, ax = plt.subplots(figsize=(6, 4))
            sns.lineplot(x=df.index, y=df[coluna_escolhida], ax=ax)
            ax.set_title("")
            st.pyplot(fig)
            st.download_button("Baixar gráfico customizado", data=salvar_grafico(fig), file_name=f"grafico_customizado_{coluna_escolhida}.png", mime="image/png", key="custom_download3")
            generated_graphs.append(coluna_escolhida)

        elif tipo_grafico == "Gráfico de Densidade" and coluna_escolhida in colunas_numericas:
            fig, ax = plt.subplots(figsize=(6, 4))
            sns.kdeplot(df[coluna_escolhida], ax=ax)
            ax.set_title("")
            st.pyplot(fig)
            st.download_button("Baixar gráfico customizado", data=salvar_grafico(fig), file_name=f"grafico_customizado_{coluna_escolhida}.png", mime="image/png", key="custom_download4")
            generated_graphs.append(coluna_escolhida)

        elif tipo_grafico == "Box Plot" and coluna_escolhida in colunas_numericas:
            fig, ax = plt.subplots(figsize=(6, 4))
            sns.boxplot(x=df[coluna_escolhida], ax=ax)
            ax.set_title("")
            st.pyplot(fig)
            st.download_button("Baixar gráfico customizado", data=salvar_grafico(fig), file_name=f"grafico_customizado_{coluna_escolhida}.png", mime="image/png", key="custom_download5")
            generated_graphs.append(coluna_escolhida)

if __name__ == "__main__":
    main()
