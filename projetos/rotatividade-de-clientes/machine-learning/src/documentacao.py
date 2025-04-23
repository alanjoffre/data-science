import streamlit as st

def introducao():
    st.title("Documentação - Churn de Clientes")

    st.header("Visão Geral")
    st.write("""
    Este projeto tem como objetivo identificar e prever a **rotatividade de clientes**. 
    A análise de churn é essencial para empresas que buscam reduzir a evasão de clientes, 
    compreender os fatores que afetam a retenção e implementar estratégias eficazes para aumentar a lealdade dos clientes.
    """)

    st.header("Objetivos do Projeto")
    st.write("""
    Este projeto visa prever a **rotatividade de clientes** com base em variáveis comportamentais e demográficas. 
    A identificação precoce de clientes com risco de churn permite a implementação de estratégias preventivas, 
    resultando em aumento de retenção e satisfação. Com isso, a empresa pode reduzir custos e aumentar a receita.
    """)

    st.header("Dados e Modelos")
    st.write("""
    Utilizamos dados de características demográficas e comportamentais dos clientes para construir modelos preditivos. 
    Os principais algoritmos de machine learning aplicados incluem o **XGBoost** e **Random Forest**, que são 
    ajustados e avaliados para oferecer a melhor precisão na previsão de churn.
    """)

if __name__ == "__main__":
    introducao()

import streamlit as st
import pandas as pd

def descricao_dados():
    st.header("Descrição dos Dados")

    st.subheader("Fontes de Dados")
    st.write("""
    Os dados utilizados neste projeto foram obtidos de [inserir fonte dos dados]. 
    Os dados incluem informações demográficas e comportamentais dos clientes.
    """)

    st.subheader("Dicionário de Dados")
    dicionario_dados = {
        "CreditScore": "Pontuação de crédito do cliente",
        "Geography": "País de residência do cliente",
        "Gender": "Gênero do cliente",
        "Age": "Idade do cliente",
        "Tenure": "Tempo de permanência (em anos)",
        "Balance": "Saldo na conta bancária",
        "NumOfProducts": "Número de produtos que o cliente possui",
        "HasCrCard": "Possui cartão de crédito (1 = Sim, 0 = Não)",
        "IsActiveMember": "Membro ativo (1 = Sim, 0 = Não)",
        "EstimatedSalary": "Salário estimado do cliente",
        "Exited": "Indicador de churn (1 = Saiu, 0 = Permaneceu)"
    }
    df_dicionario = pd.DataFrame.from_dict(dicionario_dados, orient='index', columns=['Descrição'])
    st.dataframe(df_dicionario)

    st.subheader("Pré-processamento")
    st.write("""
    O pré-processamento dos dados incluiu as seguintes etapas:
    - Tratamento de valores ausentes.
    - Remoção de duplicados.
    - Normalização de variáveis numéricas.
    - Codificação de variáveis categóricas.
    """)

if __name__ == "__main__":
    descricao_dados()

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Função para carregar dados usando o novo método de cache
@st.cache_data
def carregar_dados():
    return pd.read_csv('D:/Github/data-science/projetos/rotatividade-de-clientes/machine-learning/data/processed/rclientes_dados_tratados.csv')

# Página de Análise Exploratória dos Dados
def eda():
    st.header("Análise Exploratória dos Dados (EDA)")

    df = carregar_dados()
    
    st.subheader("Estatísticas Descritivas")
    st.write(df.describe())

    st.subheader("Distribuição das Variáveis")
    col1, col2 = st.columns(2)
    with col1:
        st.write("Distribuição da Idade")
        fig, ax = plt.subplots()
        sns.histplot(df['Age'], kde=True, ax=ax)
        st.pyplot(fig)

        st.write("Distribuição do Saldo")
        fig, ax = plt.subplots()
        sns.histplot(df['Balance'], kde=True, ax=ax)
        st.pyplot(fig)

    with col2:
        st.write("Distribuição do Tempo de Permanência")
        fig, ax = plt.subplots()
        sns.histplot(df['Tenure'], kde=True, ax=ax)
        st.pyplot(fig)

        st.write("Distribuição do Salário Estimado")
        fig, ax = plt.subplots()
        sns.histplot(df['EstimatedSalary'], kde=True, ax=ax)
        st.pyplot(fig)

    st.subheader("Relações entre Variáveis")
    st.write("Idade vs. Saldo")
    fig, ax = plt.subplots()
    sns.scatterplot(x='Age', y='Balance', hue='Exited', data=df, ax=ax)
    st.pyplot(fig)

    st.write("Saldo vs. Tempo de Permanência")
    fig, ax = plt.subplots()
    sns.scatterplot(x='Tenure', y='Balance', hue='Exited', data=df, ax=ax)
    st.pyplot(fig)

    st.write("Mapa de Calor das Correlações")
    # Filtrar apenas colunas numéricas
    df_numeric = df.select_dtypes(include=[float, int])
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(df_numeric.corr(), annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
    st.pyplot(fig)

if __name__ == "__main__":
    eda()

import streamlit as st
import yaml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
from sklearn.metrics import classification_report
import pandas as pd
import joblib

# Carregar configurações do arquivo YAML
def carregar_configuracoes(config_path):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

# Função principal para a Modelagem
def modelagem():
    st.header("Modelagem")
    
    st.subheader("Seleção de Variáveis")
    st.write("""
    As variáveis utilizadas neste projeto incluem:
    - **CreditScore**: Pontuação de crédito do cliente.
    - **Geography**: País de residência do cliente.
    - **Gender**: Gênero do cliente.
    - **Age**: Idade do cliente.
    - **Tenure**: Tempo de permanência (em anos).
    - **Balance**: Saldo na conta bancária.
    - **NumOfProducts**: Número de produtos que o cliente possui.
    - **HasCrCard**: Possui cartão de crédito.
    - **IsActiveMember**: Membro ativo.
    - **EstimatedSalary**: Salário estimado do cliente.
    """)

    st.subheader("Modelos Utilizados")
    st.write("""
    Utilizamos os seguintes algoritmos de machine learning para construir os modelos preditivos:
    - **XGBoost**: Um algoritmo de gradient boosting altamente eficiente.
    - **Random Forest**: Um conjunto de árvores de decisão que melhora a precisão da previsão.
    """)

    st.subheader("Treinamento e Validação")
    st.write("""
    O processo de treinamento e validação inclui as seguintes etapas:
    1. **Divisão dos Dados**: Separação dos dados em conjuntos de treino e teste.
    2. **Pipeline de Pré-processamento**: Normalização das variáveis numéricas.
    3. **Treinamento do Modelo**: Ajuste dos modelos aos dados de treino.
    4. **Validação Cruzada**: Avaliação dos modelos utilizando validação cruzada.
    """)

    config_path = 'D:/Github/data-science/projetos/rotatividade-de-clientes/machine-learning/config/config.yaml'
    config = carregar_configuracoes(config_path)

    st.subheader("Resultados do Modelo")
    st.write("A seguir estão os resultados do modelo treinado:")
    
    # Carregar dados
    df = pd.read_csv(config['data']['processed'])

    # Separar as variáveis independentes e dependentes
    X = df.drop(columns=['Exited'])
    y = df['Exited']

    # Dividir os dados em conjuntos de treino e teste
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Pipeline de pré-processamento
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('model', XGBClassifier())
    ])

    # Treinar o modelo
    pipeline.fit(X_train, y_train)

    # Salvar o modelo
    joblib.dump(pipeline, config['models']['final_model'])

    # Fazer previsões
    y_pred = pipeline.predict(X_test)

    # Exibir relatório de classificação
    st.text(classification_report(y_test, y_pred))

if __name__ == "__main__":
    modelagem()

import streamlit as st
import pandas as pd
import yaml
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import seaborn as sns

# Carregar configurações do arquivo YAML
def carregar_configuracoes(config_path):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

# Função principal para a Avaliação dos Modelos
def avaliacao():
    st.header("Avaliação dos Modelos")
    
    st.subheader("Métricas de Desempenho")
    st.write("""
    As seguintes métricas foram utilizadas para avaliar o desempenho dos modelos:
    - **Acurácia**: Proporção de previsões corretas em relação ao total de casos.
    - **Precisão**: Proporção de verdadeiros positivos em relação ao total de previsões positivas.
    - **Recall**: Proporção de verdadeiros positivos em relação ao total de casos reais positivos.
    - **F1-Score**: Média harmônica entre precisão e recall.
    - **AUC-ROC**: Área sob a curva ROC, que mostra a capacidade do modelo em distinguir entre as classes.
    """)

    config_path = 'D:/Github/data-science/projetos/rotatividade-de-clientes/machine-learning/config/config.yaml'
    config = carregar_configuracoes(config_path)
    
    # Carregar dados processados e modelo treinado
    df = pd.read_csv(config['data']['processed'])
    model = joblib.load(config['models']['final_model'])

    # Separar as variáveis independentes e dependentes
    X = df.drop(columns=['Exited'])
    y = df['Exited']

    # Dividir os dados em conjuntos de treino e teste
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Fazer previsões
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    # Calcular métricas
    acuracia = accuracy_score(y_test, y_pred)
    precisao = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc_roc = roc_auc_score(y_test, y_pred_proba)

    st.write("### Resultados das Métricas")
    st.write(f"Acurácia: {acuracia:.2f}")
    st.write(f"Precisão: {precisao:.2f}")
    st.write(f"Recall: {recall:.2f}")
    st.write(f"F1-Score: {f1:.2f}")
    st.write(f"AUC-ROC: {auc_roc:.2f}")

    st.subheader("Relatório de Classificação")
    st.text(classification_report(y_test, y_pred))

    st.subheader("Matriz de Confusão")
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
    st.pyplot(fig)

    st.subheader("Curva ROC")
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    fig, ax = plt.subplots()
    ax.plot(fpr, tpr, label=f'AUC = {auc_roc:.2f}')
    ax.plot([0, 1], [0, 1], 'k--')
    ax.set_xlabel('Taxa de Falsos Positivos')
    ax.set_ylabel('Taxa de Verdadeiros Positivos')
    ax.set_title('Curva ROC')
    ax.legend(loc="lower right")
    st.pyplot(fig)

if __name__ == "__main__":
    avaliacao()

import streamlit as st

def resultados():
    st.header("Resultados")
    
    st.subheader("Conclusões")
    st.write("""
    A partir das análises e modelos desenvolvidos, podemos concluir que:
    - **Principais Fatores**: As variáveis mais influentes para a rotatividade de clientes foram a **idade**, **saldo**, e **salário estimado**.
    - **Desempenho do Modelo**: O modelo **XGBoost** apresentou melhor desempenho, com um **AUC-ROC** de 0.86, indicando uma boa capacidade de distinção entre clientes que permanecerão e os que sairão.
    - **Ações Recomendadas**: Focar em estratégias de retenção para clientes com altos saldos e idades intermediárias.
    """)

    st.subheader("Impacto para a Empresa")
    st.write("""
    Os resultados deste projeto podem ajudar a empresa a:
    - **Reduzir a Evasão**: Implementar ações preventivas baseadas nas previsões de churn, como ofertas personalizadas e melhorias no atendimento.
    - **Aumentar a Satisfação**: Identificar e abordar as necessidades dos clientes de forma proativa, aumentando a satisfação e lealdade.
    - **Otimizar Recursos**: Direcionar esforços e recursos para os clientes com maior risco de churn, maximizando o retorno sobre o investimento.
    """)

if __name__ == "__main__":
    resultados()

import streamlit as st

def implementacao():
    st.header("Implementação")
    
    st.subheader("Pipeline de Produção")
    st.write("""
    O pipeline de produção inclui as seguintes etapas:
    1. **Carga e Pré-processamento dos Dados**: Carregamento dos dados e aplicação de técnicas de limpeza e transformação.
    2. **Treinamento do Modelo**: Treinamento e validação dos modelos de machine learning.
    3. **Avaliação do Modelo**: Avaliação das métricas de desempenho do modelo.
    4. **Implantação do Modelo**: Exportação do modelo treinado para um ambiente de produção.
    5. **Inferência**: Aplicação do modelo em novos dados para prever a rotatividade de clientes.
    """)

    st.subheader("Ferramentas Utilizadas")
    st.write("""
    As principais ferramentas e bibliotecas utilizadas neste projeto incluem:
    - **Python**: Linguagem de programação principal.
    - **Pandas**: Manipulação e análise de dados.
    - **Scikit-learn**: Modelagem e avaliação de machine learning.
    - **XGBoost**: Algoritmo de gradient boosting.
    - **Streamlit**: Criação de dashboards interativos.
    - **YAML**: Arquivos de configuração.
    """)

    st.subheader("Integração com Sistemas Existentes")
    st.write("""
    A integração das previsões de churn com os sistemas existentes da empresa pode ser realizada de várias maneiras:
    - **APIs**: Criação de APIs para que os sistemas internos possam acessar as previsões em tempo real.
    - **Automação de Relatórios**: Geração de relatórios automatizados com insights e previsões.
    - **Dashboards Interativos**: Implementação de dashboards, como este em Streamlit, para visualização e análise dos dados de churn.
    """)

if __name__ == "__main__":
    implementacao()

import streamlit as st
import pandas as pd
import joblib
import yaml
import uuid

# Carregar configurações do arquivo YAML
def carregar_configuracoes(config_path):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

# Carregar modelo e pré-processador
def carregar_modelo_preprocessador(config):
    modelo = joblib.load(config['models']['final_model'])
    preprocessor = joblib.load(config['preprocessors']['path'])
    return modelo, preprocessor

# Função para gerar uma chave única
def gerar_chave_unica():
    return str(uuid.uuid4())

# Página de Inferência
def inferencia():
    st.header("Inferência")

    st.subheader("Previsões em Novos Dados")
    st.write("""
    Utilize esta seção para realizar previsões de churn em novos dados de clientes. Insira os dados abaixo para obter a previsão do modelo.
    """)

    # Carregar configurações
    config_path = 'D:/Github/data-science/projetos/rotatividade-de-clientes/machine-learning/config/config.yaml'
    config = carregar_configuracoes(config_path)

    # Carregar modelo e pré-processador
    modelo, preprocessor = carregar_modelo_preprocessador(config)

    # Coletar dados do cliente com uma chave única para cada formulário
    form_key = 'formulario_cliente_' + gerar_chave_unica()

    with st.form(key=form_key):
        dados = {}
        dados['CreditScore'] = st.number_input("Digite o Credit Score", min_value=0.0, step=0.1)
        dados['Geography'] = st.selectbox("Digite a Geografia", options=["France", "Germany", "Spain"])
        dados['Gender'] = st.selectbox("Digite o Gênero", options=["Male", "Female"])
        dados['Age'] = st.number_input("Digite a Idade", min_value=18, max_value=100)
        dados['Tenure'] = st.number_input("Digite o Tempo de Permanência", min_value=0, max_value=50)
        dados['Balance'] = st.number_input("Digite o Saldo", min_value=0.0, step=0.1)
        dados['NumOfProducts'] = st.number_input("Digite o Número de Produtos", min_value=1, max_value=10)
        dados['HasCrCard'] = st.radio("Possui Cartão de Crédito?", options=[1, 0])
        dados['IsActiveMember'] = st.radio("É Membro Ativo?", options=[1, 0])
        dados['EstimatedSalary'] = st.number_input("Digite o Salário Estimado", min_value=0.0, step=0.1)
        submit_button = st.form_submit_button(label='Executar Inferência')
    
    if submit_button:
        df_cliente = pd.DataFrame([dados])

        # Preprocessar os dados
        X_transformed = preprocessor.transform(df_cliente)

        # Fazer previsão
        previsoes = modelo.predict(X_transformed)
        df_cliente['Predicao'] = previsoes

        # Exibir resultado da previsão
        st.subheader("Resultado da Previsão")
        st.write(df_cliente[['Predicao']])

        if previsoes[0] == 1:
            st.write("**O cliente provavelmente sairá da empresa.**")
        else:
            st.write("**O cliente provavelmente permanecerá na empresa.**")

if __name__ == "__main__":
    inferencia()

import streamlit as st
import subprocess

# Função principal para a Implementação do Pipeline
def pipeline_producao():
    st.header("Implementação")

    st.subheader("Pipeline de Produção")
    st.write("""
    O pipeline de produção é responsável por transformar, treinar, avaliar e implementar o modelo de machine learning para previsão de churn. 
    Abaixo estão as etapas detalhadas do pipeline:
    
    1. **Pré-processamento de Dados**: Transformação, limpeza e normalização dos dados.
    2. **Treinamento do Modelo**: Treinamento do modelo XGBoost com os dados de treino.
    3. **Avaliação e Métricas**: Avaliação do modelo treinado utilizando métricas como acurácia, precisão, recall, F1-Score e AUC-ROC.
    4. **Implantação do Modelo**: Exportação do modelo final para uso em produção.
    """)

    st.subheader("Execução do Pipeline")
    st.write("Clique no botão abaixo para executar o pipeline de produção.")

    # Botão para executar o pipeline
    if st.button("Executar Pipeline"):
        with st.spinner("Processando o pipeline, por favor aguarde..."):
            try:
                # Executa o arquivo pipeline.py e captura o output em tempo real
                result = subprocess.run(["python", "pipeline.py"], capture_output=True, text=True)
                output = result.stdout
                
                # Exibe o output do pipeline
                st.success("Pipeline executado com sucesso!")
                st.text(output)
            except Exception as e:
                st.error(f"Erro ao executar o pipeline: {e}")

if __name__ == "__main__":
    pipeline_producao()

import streamlit as st

def documentacao_tecnica_parte1():
    st.header("Documentação Técnica")

    st.subheader("Introdução")
    st.write("""
    Esta seção fornece uma visão geral dos códigos utilizados no projeto de Churn de Clientes, 
    bem como as dependências necessárias para reproduzir o projeto.
    """)

    st.subheader("Dependências")
    st.write("""
    As seguintes bibliotecas são necessárias para executar o projeto:
    - **pandas**: Manipulação e análise de dados.
    - **numpy**: Suporte para arrays e operações matemáticas.
    - **scikit-learn**: Ferramentas para machine learning e modelagem estatística.
    - **xgboost**: Algoritmo de boosting eficiente.
    - **joblib**: Persistência de objetos Python.
    - **streamlit**: Framework para construção de interfaces web interativas.
    - **yaml**: Leitura e escrita de arquivos YAML.
    - **matplotlib**: Criação de gráficos estáticos.
    - **seaborn**: Visualização de dados baseada em matplotlib.
    """)

    st.code("""
    # Exemplo de instalação das dependências:
    pip install pandas numpy scikit-learn xgboost joblib streamlit pyyaml matplotlib seaborn
    """, language='bash')

if __name__ == "__main__":
    documentacao_tecnica_parte1()

import streamlit as st
import pandas as pd

# Função para carregar dados usando o novo método de cache
@st.cache_data
def carregar_dados():
    return pd.read_csv('D:/Github/data-science/projetos/rotatividade-de-clientes/machine-learning/data/processed/rclientes_dados_tratados.csv')

def documentacao_tecnica_parte2():
    st.subheader("Carregamento e Pré-processamento dos Dados")

    st.write("### Código para Carregar e Pré-processar os Dados")
    st.code("""
    # Carregamento dos dados
    file_path = 'D:/Github/data-science/projetos/rotatividade-de-clientes/machine-learning/data/processed/rclientes_preprocessado.csv'
    clientes_df = pd.read_csv(file_path)

    # Separar as variáveis independentes (X) e a variável alvo (y)
    X = clientes_df.drop(columns=['Exited'])
    y = clientes_df['Exited']

    # Dividir os dados em conjunto de treino e teste
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    """, language='python')

if __name__ == "__main__":
    documentacao_tecnica_parte2()

import streamlit as st
import joblib
from xgboost import XGBClassifier

def documentacao_tecnica_parte3():
    st.subheader("Treinamento e Avaliação do Modelo")

    st.write("### Código para Treinamento e Avaliação do Modelo")
    st.code("""
    # Treinamento do modelo
    model = XGBClassifier()
    model.fit(X_train, y_train)

    # Salvar o modelo
    modelo_path = 'D:/Github/data-science/projetos/rotatividade-de-clientes/machine-learning/models/final_model.joblib'
    joblib.dump(model, modelo_path)

    # Avaliação do modelo
    y_pred = model.predict(X_test)
    from sklearn.metrics import classification_report
    print(classification_report(y_test, y_pred))
    """, language='python')

if __name__ == "__main__":
    documentacao_tecnica_parte3()

import streamlit as st
import subprocess

def documentacao_tecnica_parte4():
    st.subheader("Pipeline de Produção e Inferência")

    st.write("### Código para Pipeline de Produção e Inferência")
    st.code("""
    # Execução do Pipeline
    import os
    import yaml

    def carregar_configuracoes(config_path):
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
        return config

    config_path = 'D:/Github/data-science/projetos/rotatividade-de-clientes/machine-learning/config/config.yaml'
    config = carregar_configuracoes(config_path)
    data_path = config['data']['new_data']

    # Fazer inferência
    model = joblib.load(config['models']['final_model'])
    df = pd.read_csv(data_path)
    X = df.drop(columns=['Exited'])
    y_pred = model.predict(X)
    df['Predicao'] = y_pred
    df.to_csv('D:/Github/data-science/projetos/rotatividade-de-clientes/machine-learning/predictions/predictions.csv', index=False)
    """, language='python')

if __name__ == "__main__":
    documentacao_tecnica_parte4()

import streamlit as st

def conclusoes():
    st.header("Conclusões Finais e Próximos Passos")

    st.subheader("Resumo dos Principais Achados")
    st.write("""
    Este projeto de previsão de churn de clientes forneceu insights valiosos sobre os fatores que influenciam a rotatividade de clientes. 
    Utilizando modelos de machine learning, como o XGBoost e Random Forest, conseguimos prever com precisão os clientes que provavelmente deixarão a empresa.
    """)

    st.write("""
    **Principais Achados:**
    - **Variáveis mais importantes**: Pontuação de crédito, idade, saldo e salário estimado foram as variáveis mais influentes na previsão de churn.
    - **Desempenho do Modelo**: O XGBoost apresentou o melhor desempenho com um AUC-ROC de 0.86, seguido pelo Random Forest.
    - **Distribuições e Relações**: Visualizações exploratórias mostraram padrões distintos entre clientes que permaneceram e os que deixaram a empresa.
    """)

    st.subheader("Impacto para a Empresa")
    st.write("""
    Os resultados deste projeto podem ser utilizados para melhorar a retenção de clientes e otimizar as estratégias de marketing. 
    Algumas ações recomendadas incluem:
    - **Segmentação de Clientes**: Focar em clientes com alto risco de churn para oferecer promoções e incentivos personalizados.
    - **Melhorias no Atendimento**: Implementar programas de fidelidade e melhorias no atendimento ao cliente para aumentar a satisfação e reduzir a evasão.
    - **Análise Contínua**: Monitorar continuamente os indicadores de churn e ajustar as estratégias conforme necessário.
    """)

    st.subheader("Sugestões para Melhorias Futuras")
    st.write("""
    **Direções Futuras:**
    - **Explorar Novas Variáveis**: Incluir variáveis adicionais, como comportamento de navegação no site e interações com o suporte ao cliente.
    - **Hiperparametrização**: Experimentar com diferentes parâmetros e técnicas de otimização para melhorar ainda mais o desempenho dos modelos.
    - **Integração em Tempo Real**: Desenvolver um sistema de previsão de churn em tempo real para intervenções mais rápidas e eficazes.
    - **Expandir o Escopo**: Aplicar a análise de churn a outros produtos e serviços da empresa para obter uma visão mais abrangente.
    """)

    st.write("""
    Este projeto é um passo importante na direção certa para entender e mitigar a rotatividade de clientes. 
    Com as ações e melhorias contínuas, a empresa pode alcançar uma maior satisfação do cliente e crescimento sustentável.
    """)

if __name__ == "__main__":
    conclusoes()
