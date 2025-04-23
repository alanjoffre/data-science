import streamlit as st
import pandas as pd
import joblib
import logging
import yaml
import os

# Configurar logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Caminho para as configurações e modelos
config_path = os.path.join('D:/Github/data-science/projetos/rotatividade-de-clientes/machine-learning/config/config.yaml')

# Função para carregar as configurações do arquivo YAML
def carregar_configuracoes(config_path):
    """Carrega as configurações do arquivo YAML."""
    try:
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
            logger.info(f"Configurações carregadas de: {config_path}")
            return config
    except Exception as e:
        logger.error(f"Erro ao carregar o arquivo de configurações: {e}")
        raise

# Função para carregar o modelo salvo
def carregar_modelo(model_path):
    """Carrega o modelo salvo."""
    try:
        modelo = joblib.load(model_path)
        logger.info(f"Modelo carregado de: {model_path}")
        return modelo
    except Exception as e:
        logger.error(f"Erro ao carregar o modelo: {e}")
        raise

# Função para carregar o pré-processador salvo
def carregar_preprocessador(preprocessor_path):
    """Carrega o pré-processador salvo."""
    try:
        preprocessor = joblib.load(preprocessor_path)
        logger.info(f"Pré-processador carregado de: {preprocessor_path}")
        return preprocessor
    except Exception as e:
        logger.error(f"Erro ao carregar o pré-processador: {e}")
        raise

# Função para coletar dados do cliente via Streamlit
def coletar_dados_cliente():
    """Coleta dados do cliente via entrada do usuário no Streamlit."""
    dados = {}
    with st.form(key='formulario_cliente'):  # Formulário para agrupar os campos
        dados['CreditScore'] = st.number_input("Digite o Credit Score", min_value=0.0, step=0.1)
        dados['Geography'] = st.selectbox("Digite a Geografia", options=["France", "Germany", "Spain"])  # Exemplo
        dados['Gender'] = st.selectbox("Digite o Gênero", options=["Male", "Female"])
        dados['Age'] = st.number_input("Digite a Idade", min_value=18, max_value=100)
        dados['Tenure'] = st.number_input("Digite o Tempo de Permanência", min_value=0, max_value=50)
        dados['Balance'] = st.number_input("Digite o Saldo", min_value=0.0, step=0.1)
        dados['NumOfProducts'] = st.number_input("Digite o Número de Produtos", min_value=1, max_value=10)
        dados['HasCrCard'] = st.radio("Possui Cartão de Crédito?", options=[1, 0])
        dados['IsActiveMember'] = st.radio("É Membro Ativo?", options=[1, 0])
        dados['EstimatedSalary'] = st.number_input("Digite o Salário Estimado", min_value=0.0, step=0.1)

        submit_button = st.form_submit_button(label='Executar Inferência')  # Botão dentro do formulário
    
    if submit_button:
        return pd.DataFrame([dados])
    else:
        return None

# Função para fazer a inferência com o modelo e o pré-processador carregados
def fazer_inferencia(config):
    """Faz a inferência utilizando o modelo e pré-processador carregados."""
    # Carregar o modelo e o pré-processador a partir da configuração
    modelo = carregar_modelo(config['models']['final_model'])
    preprocessor = carregar_preprocessador(config['preprocessors']['path'])

    # Coletar os dados do cliente
    df_cliente = coletar_dados_cliente()

    if df_cliente is not None:
        # Preprocessar os dados
        X_transformed = preprocessor.transform(df_cliente)

        # Fazer a previsão
        try:
            previsoes = modelo.predict(X_transformed)
            df_cliente['Predicao'] = previsoes
            logger.info("Inferência concluída com sucesso.")
            
            # Exibir o resultado da previsão e a mensagem personalizada
            st.subheader("Resultado da previsão:")
            st.write(df_cliente[['Predicao']])

            # Lógica personalizada para a previsão
            if previsoes[0] == 1:
                st.write("**O cliente provavelmente sairá da empresa.**")
            else:
                st.write("**O cliente provavelmente permanecerá na empresa.**")
            
        except Exception as e:
            logger.error(f"Erro ao fazer a inferência: {e}")
            st.error(f"Erro ao fazer a inferência: {e}")
    else:
        st.info("Por favor, preencha todos os campos e pressione o botão para fazer a inferência.")

# Função principal para exibir a página de inferência no Streamlit
def main():
    # Definindo o estilo para o título e fundo da página
    st.markdown(""" 
    <style>
        .pipeline-title {
            background-color: #ff4b4b;
            color: white;
            padding: 10px;
            border-radius: 5px;
            text-align: center;
            font-size: 24px;
            margin-bottom: 20px; /* Adicionando margem inferior */
        }
        .section-title {
            font-size: 18px;
            font-weight: bold;
            margin-top: 10px;
            margin-bottom: 10px;
        }
        .content-text {
            font-size: 14px;
        }
    </style>
    <div class="pipeline-title">Inferência do Modelo de Rotatividade de Clientes</div>
    """, unsafe_allow_html=True)

    # Carregar configurações
    config = carregar_configuracoes(config_path)

    # Título da seção de inferência
    st.markdown("<h3 style='margin-bottom: 30px;'>Preencha os dados abaixo para fazer a previsão:</h3>", unsafe_allow_html=True)

    # Rodar inferência com o formulário
    fazer_inferencia(config)

if __name__ == "__main__":
    main()
