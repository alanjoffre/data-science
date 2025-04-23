import sys
import os
import time
import psutil
import logging
import yaml
from datetime import datetime
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, roc_auc_score
from xgboost import XGBClassifier
import joblib
from sklearn.model_selection import train_test_split
import pandas as pd  # Importando pandas para carregar os dados

# Adicionar o diretório 'src' ao sys.path
sys.path.append(os.path.join('D:\\Github\\data-science\\projetos\\rotatividade-de-clientes\\machine-learning\\src'))

# Definir o diretório de logs e o nome do arquivo de log com data e hora da execução
log_dir = 'D:\\Github\\data-science\\projetos\\rotatividade-de-clientes\\machine-learning\\logs'
log_file_path = os.path.join(log_dir, f"pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")

# Garantir que o diretório de logs exista
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

# Configuração do logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Configurar o FileHandler para o arquivo de log e o StreamHandler para o console
file_handler = logging.FileHandler(log_file_path, mode='a')
file_handler.setLevel(logging.INFO)
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)

# Definir o formato dos logs
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
console_handler.setFormatter(formatter)

# Adicionar os handlers ao logger
logger.addHandler(file_handler)
logger.addHandler(console_handler)

# Função para monitorar a latência e o uso de recursos
def monitorar_performance(etapa):
    """Monitora o uso de recursos do sistema (CPU e memória)."""
    try:
        cpu_usage = psutil.cpu_percent(interval=1)
        memory_usage = psutil.virtual_memory().percent
        logger.info(f"[{etapa}] Uso de CPU: {cpu_usage}%")
        logger.info(f"[{etapa}] Uso de memória: {memory_usage}%")
    except Exception as e:
        logger.error(f"Erro ao monitorar performance na etapa {etapa}: {e}")

# Função para calcular e registrar as métricas de performance do modelo
def avaliar_modelo(modelo, X_test, y_test):
    """Avalia o modelo usando métricas de classificação e registra as métricas no log."""
    try:
        # Fazer predições
        y_pred = modelo.predict(X_test)
        y_prob = modelo.predict_proba(X_test)[:, 1]

        # Calcular métricas
        accuracy = accuracy_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_prob)

        # Registrar as métricas no log
        logger.info(f"Métricas de desempenho do modelo:")
        logger.info(f"Acurácia: {accuracy:.4f}")
        logger.info(f"Recall: {recall:.4f}")
        logger.info(f"Precisão: {precision:.4f}")
        logger.info(f"F1-Score: {f1:.4f}")
        logger.info(f"AUC: {auc:.4f}")
    except Exception as e:
        logger.error(f"Erro ao avaliar o modelo: {e}")

# Função para carregar a configuração do arquivo YAML
def carregar_configuracao(config_path='D:\\Github\\data-science\\projetos\\rotatividade-de-clientes\\machine-learning\\config\\config.yaml'):
    """Carregar configurações de um arquivo YAML."""
    try:
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
        logger.info("Configuração carregada com sucesso.")
        return config
    except FileNotFoundError:
        logger.error(f"Arquivo de configuração não encontrado: {config_path}")
        raise
    except yaml.YAMLError as e:
        logger.error(f"Erro ao ler o arquivo de configuração: {e}")
        raise

# Função de pré-processamento de dados
def preprocessamento_dados():
    """Processa os dados e retorna X_train, X_test, y_train e y_test."""
    try:
        # Carregar dados e realizar o pré-processamento
        dados = pd.read_csv('D:\\Github\\data-science\\projetos\\rotatividade-de-clientes\\machine-learning\\data\\processed\\rclientes_preprocessado.csv')

        # Separar as variáveis independentes (X) e dependentes (y)
        X = dados.drop('Exited', axis=1)  # Ajuste conforme seu dataset
        y = dados['Exited']  # Ajuste conforme seu dataset

        # Dividir os dados em treino e teste
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        return X_train, X_test, y_train, y_test
    except Exception as e:
        logger.error(f"Erro ao processar os dados: {e}")
        raise

# Função para treinar o modelo
def treinar_modelo(X_train, y_train):
    """Treina o modelo XGBoost e retorna o modelo treinado."""
    try:
        # Treinamento do modelo
        modelo = XGBClassifier()
        modelo.fit(X_train, y_train)
        return modelo
    except Exception as e:
        logger.error(f"Erro ao treinar o modelo: {e}")
        raise

# Função para salvar o modelo treinado
def salvar_modelo(modelo, caminho):
    """Salva o modelo treinado em um arquivo."""
    try:
        joblib.dump(modelo, caminho)
        logger.info(f"Modelo salvo com sucesso em {caminho}.")
    except Exception as e:
        logger.error(f"Erro ao salvar o modelo: {e}")
        raise

# Função para executar o pipeline
def executar_pipeline():
    """Executar o pipeline de pré-processamento, treinamento e avaliação do modelo."""
    try:
        # Carregar configuração
        config = carregar_configuracao()
        
        # Iniciar pré-processamento
        logger.info("Iniciando o pipeline de pré-processamento...")
        monitorar_performance("Pré-processamento (início)")
        X_train, X_test, y_train, y_test = preprocessamento_dados()  # Chama a função de pré-processamento
        monitorar_performance("Pré-processamento (fim)")

        # Iniciar treinamento do modelo
        logger.info("Iniciando o treinamento do modelo...")
        monitorar_performance("Treinamento (início)")
        modelo = treinar_modelo(X_train, y_train)  # Chama a função de treinamento e obtém o modelo
        monitorar_performance("Treinamento (fim)")

        # Avaliar e registrar as métricas de performance do modelo
        logger.info("Avaliando o modelo...")
        avaliar_modelo(modelo, X_test, y_test)

        # Salvando o modelo treinado
        logger.info("Salvando o modelo...")
        salvar_modelo(modelo, config['models']['final_model'])

        logger.info("Pipeline concluído com sucesso.")
    except Exception as e:
        logger.error(f"Erro durante a execução do pipeline: {e}", exc_info=True)

if __name__ == "__main__":
    logger.info("Iniciando a execução do pipeline.")
    
    executar_pipeline()

    # Remover handlers e fechar o FileHandler para garantir a gravação no arquivo
    logger.removeHandler(file_handler)
    file_handler.close()
