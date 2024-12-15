import pandas as pd
import joblib
import logging
import yaml
import os

# Configurar logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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

def fazer_inferencia(data_path, model_path, preprocessor_path):
    """Faz a inferência utilizando o modelo e pré-processador carregados."""
    # Carregar dados de entrada
    try:
        df = pd.read_csv(data_path)
        logger.info(f'Dados de entrada carregados de: {data_path}')
    except Exception as e:
        logger.error(f"Erro ao carregar os dados de entrada: {e}")
        raise

    # Carregar modelo e pré-processador
    modelo = carregar_modelo(model_path)
    preprocessor = carregar_preprocessador(preprocessor_path)

    # Preprocessar os dados
    X = df.drop('Exited', axis=1) if 'Exited' in df.columns else df
    X_transformed = preprocessor.transform(X)

    # Fazer previsão
    try:
        previsoes = modelo.predict(X_transformed)
        df['Predicao'] = previsoes
        logger.info("Inferência concluída com sucesso.")
    except Exception as e:
        logger.error(f"Erro ao fazer a inferência: {e}")
        raise

    # Salvar resultados
    resultados_path = os.path.join(config['predictions']['directory'], 'resultados_inferencia.csv')
    try:
        df.to_csv(resultados_path, index=False)
        logger.info(f'Resultados da inferência salvos em: {resultados_path}')
    except Exception as e:
        logger.error(f"Erro ao salvar os resultados da inferência: {e}")
        raise

if __name__ == "__main__":
    config_path = os.path.join('D:/Github/data-science/projetos/rotatividade-de-clientes/machine-learning/config/config.yaml')
    config = carregar_configuracoes(config_path)

    fazer_inferencia(
        data_path=config['data']['new_data'],  # Usando o caminho de dados novos
        model_path=config['models']['final_model'],
        preprocessor_path=config['preprocessors']['path']
    )