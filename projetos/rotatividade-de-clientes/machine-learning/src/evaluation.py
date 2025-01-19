import joblib
import pandas as pd
import yaml
import logging
import numpy as np
from sklearn.metrics import classification_report, roc_curve, auc, accuracy_score, f1_score
from sklearn.model_selection import train_test_split

# Configurar o logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def carregar_configuracao(config_path='D:/Github/data-science/projetos/rotatividade-de-clientes/machine-learning/config/config.yaml'):
    """
    Carregar configurações de um arquivo YAML.

    Parameters:
    config_path (str): O caminho para o arquivo de configuração YAML.

    Returns:
    dict: Um dicionário contendo as configurações carregadas do arquivo.

    Raises:
    Exception: Se houver um erro ao carregar as configurações.
    """
    try:
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
        logger.info("Configuração carregada com sucesso.")
        return config
    except Exception as e:
        logger.error(f"Erro ao carregar configuração: {e}")
        raise

def carregar_dados_e_modelo(config):
    """
    Carregar dados e modelo treinado a partir dos caminhos especificados no arquivo de configuração.

    Parameters:
    config (dict): Dicionário contendo as configurações do projeto, incluindo os caminhos dos dados processados e do modelo treinado.

    Returns:
    tuple: DataFrame contendo os dados carregados e o modelo treinado.

    Raises:
    Exception: Se houver um erro ao carregar os dados ou o modelo.
    """
    try:
        df = pd.read_csv(config['data']['processed'])
        model = joblib.load(config['models']['final_model'])
        logger.info("Dados e modelo carregados com sucesso.")
        return df, model
    except Exception as e:
        logger.error(f"Erro ao carregar dados ou modelo: {e}")
        raise

def avaliar_modelo(model, X_test, y_test, config):
    """
    Avaliar o modelo usando métricas de classificação e salvar os resultados.

    Parameters:
    model (object): O modelo treinado a ser avaliado.
    X_test (pd.DataFrame): Conjunto de características de teste.
    y_test (pd.Series): Conjunto de etiquetas de teste.
    config (dict): Dicionário contendo as configurações do projeto, incluindo os caminhos para salvar os relatórios.

    Returns:
    None

    Raises:
    Exception: Se houver um erro durante a avaliação do modelo.
    """
    try:
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

        report = classification_report(y_test, y_pred)
        logger.info("Relatório de Classificação:\n%s", report)

        # Salvar relatório em TXT
        report_path = config['reports']['classification_final']
        with open(report_path, 'w') as f:
            f.write(report)
        logger.info(f"Relatório de classificação salvo em: {report_path}")

        fpr, tpr, thresholds = roc_curve(y_test, y_prob)
        roc_auc = auc(fpr, tpr)
        optimal_idx = np.argmax(tpr - fpr)
        optimal_threshold = thresholds[optimal_idx]
        logger.info(f"AUC da curva ROC: {roc_auc:.4f}")
        logger.info(f"Melhor threshold encontrado: {optimal_threshold:.4f}")

        y_pred_optimal = (y_prob >= optimal_threshold).astype(int)
        report_optimal = classification_report(y_test, y_pred_optimal)
        logger.info("Relatório com Threshold Otimizado:\n%s", report_optimal)

        report_threshold_path = config['reports']['classification_threshold']
        with open(report_threshold_path, 'w') as f:
            f.write(report_optimal)
        logger.info(f"Relatório de classificação com threshold otimizado salvo em: {report_threshold_path}")
    except Exception as e:
        logger.error(f"Erro ao avaliar o modelo: {e}")
        raise

def main():
    """
    Função principal para executar a avaliação do modelo.

    Esta função executa as seguintes etapas:
    1. Carrega as configurações do arquivo YAML.
    2. Carrega os dados processados e o modelo treinado.
    3. Divide os dados em conjuntos de treino e teste.
    4. Avalia o modelo usando métricas de classificação e salva os resultados.

    Raises:
    Exception: Se houver um erro em qualquer etapa da avaliação.
    """
    try:
        config = carregar_configuracao()
        df, model = carregar_dados_e_modelo(config)
        
        # Dividir os dados em conjuntos de treino e teste
        X = df.drop('Exited', axis=1)
        y = df['Exited']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Avaliar o modelo
        avaliar_modelo(model, X_test, y_test, config)
    except Exception as e:
        logger.error(f"Erro durante a execução do script: {e}")
        raise

if __name__ == "__main__":
    main()