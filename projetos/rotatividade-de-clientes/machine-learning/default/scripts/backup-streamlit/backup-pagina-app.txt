import streamlit as st
import yaml
from streamlit_option_menu import option_menu
from page_dashboard import main as dashboard_main  # Importando a função do arquivo correto
from page_pipeline import main as pipeline_main  # Importando a função do novo arquivo para o pipeline
from page_avaliacao import main as avaliacao_main  # Importando a função para a página de avaliação
from page_inferencia_input import main as previsao_main  # Corrigido para importar a função de previsao_main

# Carregar configurações do arquivo YAML
def carregar_config():
    with open('D:/Github/data-science/projetos/rotatividade-de-clientes/machine-learning/config/config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    return config

# Carregar configurações do projeto
config = carregar_config()

# Centralizar o título "Projeto: Churn de Clientes" com fundo destacado (apenas no início)
st.markdown("""
<style>
    .title {
        background-color: #ff4b4b;
        color: white;
        padding: 10px;
        border-radius: 5px;
        text-align: center;
        font-size: 32px;
        margin-top: 20px;
    }
</style>
<div class="title">Projeto: Churn de Clientes</div>
""", unsafe_allow_html=True)

# Visão geral do projeto
st.write("### Visão Geral")
st.write(
    "Este projeto tem como objetivo identificar e prever a **rotatividade de clientes**. "
    "A análise de churn é essencial para empresas que buscam reduzir a evasão de clientes, "
    "compreender os fatores que afetam a retenção e implementar estratégias eficazes para aumentar a lealdade dos clientes."
)

# Objetivos do Projeto
st.write("### Objetivos do Projeto")
st.write(
    "Este projeto visa prever a **rotatividade de clientes** com base em variáveis comportamentais e demográficas. "
    "A identificação precoce de clientes com risco de churn permite a implementação de estratégias preventivas, "
    "resultando em aumento de retenção e satisfação. Com isso, a empresa pode reduzir custos e aumentar a receita."
)

# Dados e Modelos
st.write("### Dados e Modelos")
st.write(
    "Utilizamos dados de características demográficas e comportamentais dos clientes para construir modelos preditivos. "
    "Os principais algoritmos de machine learning aplicados incluem o **XGBoost** e **Random Forest**, que são "
    "ajustados e avaliados para oferecer a melhor precisão na previsão de churn."
)

# Configuração da barra lateral para navegação do menu com título
with st.sidebar:
    st.image("D:/Github/data-science/projetos/rotatividade-de-clientes/machine-learning/reports/figures/churn_logo.png", width=300)
    st.markdown("## Navegação")
    menu = option_menu(
        menu_title="",
        options=["Página Inicial", "Dashboard", "Pipeline de Produção", "Avaliação", "Inferência", "Previsão de Clientes"],
        icons=["house", "bar-chart-line", "gear", "clipboard-data", "search", "person-plus"],
        menu_icon="menu-app-fill",
        default_index=0,
        orientation="vertical",
    )

# Navegação entre seções com base na seleção do menu
if menu == "Página Inicial":
    st.write("### Navegue pelo Projeto")
    st.write(
        "Use o menu ao lado para explorar as diferentes seções do projeto:\n"
        "- **Dashboard**: Visualize as métricas e gráficos que explicam o comportamento dos clientes.\n"
        "- **Pipeline de Produção**: Entenda o fluxo de dados e as etapas do pipeline.\n"
        "- **Avaliação**: Explore os resultados dos modelos e as métricas de desempenho.\n"
        "- **Inferência**: Realize previsões em novos dados.\n"
        "- **Previsão de Clientes**: Preveja a probabilidade de churn de novos clientes."
    )
elif menu == "Dashboard":
    dashboard_main()  # Chama a função principal do page_dashboard.py
elif menu == "Pipeline de Produção":
    pipeline_main()  # Chama a função principal do page_pipeline.py, que foi criada com Streamlit para o pipeline
elif menu == "Avaliação":
    avaliacao_main()  # Chama a função principal do page_avaliacao.py para executar a avaliação e exibir resultados
elif menu == "Inferência":
    inferencia_main()  # Chama a função principal do page_inferencia.py para executar a inferência e exibir resultados
elif menu == "Previsão de Clientes":
    previsao_main()  # Chama a função principal de page_inferencia_input.py para previsão de churn

# Barra de progresso
progresso = 1.00
st.write("### Status do Projeto")
st.progress(progresso)
st.write(f"Fase atual: **Modelagem e Avaliação** - {progresso*100}% completo")

# Links úteis e rodapé
st.write("### Links Úteis")
st.markdown("[Repositório GitHub](https://github.com/alanjoffre/data-science/tree/master/projetos/rotatividade-de-clientes)")
st.markdown("[Documentação do Projeto](https://link-para-documentacao.com)")
st.markdown("---")
st.markdown("<h5 style='text-align: center;'>Desenvolvido por: Alan Joffre - 2024</h5>", unsafe_allow_html=True)
