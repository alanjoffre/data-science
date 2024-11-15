import pandas as pd
import joblib
import logging
import yaml
import os

# Configurar logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Carregar configurações do arquivo YAML
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

def carregar_modelo(model_path):
    """Carrega o modelo salvo."""
    try:
        modelo = joblib.load(model_path)
        logger.info(f"Modelo carregado de: {model_path}")
        return modelo
    except Exception as e:
        logger.error(f"Erro ao carregar o modelo: {e}")
        raise

def carregar_preprocessador(preprocessor_path):
    """Carrega o pré-processador salvo."""
    try:
        preprocessor = joblib.load(preprocessor_path)
        logger.info(f"Pré-processador carregado de: {preprocessor_path}")
        return preprocessor
    except Exception as e:
        logger.error(f"Erro ao carregar o pré-processador: {e}")
        raise

def coletar_dados_cliente():
    """Coleta dados do cliente via entrada do usuário."""
    dados = {}
    dados['CreditScore'] = float(input("Digite o Credit Score: "))
    dados['Geography'] = input("Digite a Geografia: ")
    dados['Gender'] = input("Digite o Gênero (Male/Female): ")
    dados['Age'] = int(input("Digite a Idade: "))
    dados['Tenure'] = int(input("Digite o Tempo de Permanência: "))
    dados['Balance'] = float(input("Digite o Saldo: "))
    dados['NumOfProducts'] = int(input("Digite o Número de Produtos: "))
    dados['HasCrCard'] = int(input("Possui Cartão de Crédito? (1-Sim, 0-Não): "))
    dados['IsActiveMember'] = int(input("É Membro Ativo? (1-Sim, 0-Não): "))
    dados['EstimatedSalary'] = float(input("Digite o Salário Estimado: "))
    return pd.DataFrame([dados])

def fazer_inferencia():
    """Faz a inferência utilizando o modelo e pré-processador carregados."""
    # Carregar modelo e pré-processador
    modelo = carregar_modelo(config['models']['final_model'])
    preprocessor = carregar_preprocessador(config['preprocessors']['path'])

    # Coletar dados do cliente
    df_cliente = coletar_dados_cliente()

    # Preprocessar os dados
    X_transformed = preprocessor.transform(df_cliente)

    # Fazer previsão
    try:
        previsoes = modelo.predict(X_transformed)
        df_cliente['Predicao'] = previsoes
        logger.info("Inferência concluída com sucesso.")
        print("Resultado da previsão:", df_cliente[['Predicao']])
    except Exception as e:
        logger.error(f"Erro ao fazer a inferência: {e}")
        raise

if __name__ == "__main__":
    config_path = os.path.join('D:/Github/data-science/projetos/rotatividade-de-clientes/machine-learning/config/config.yaml')
    config = carregar_configuracoes(config_path)
    fazer_inferencia()