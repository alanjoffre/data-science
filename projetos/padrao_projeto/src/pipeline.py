import os
import yaml
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from tslearn.clustering import TimeSeriesKMeans, KShape
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
import mlflow
import optuna
from statsmodels.tsa.seasonal import STL

# Configuração do Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

#############################
# Funções de Ingestão e Configuração
#############################
def load_config(config_path: str) -> dict:
    """Carrega e retorna o conteúdo do arquivo YAML de configuração."""
    try:
        with open(config_path, 'r', encoding='utf-8') as file:
            config = yaml.safe_load(file)
        logger.info("Configuração carregada com sucesso.")
        return config
    except Exception as e:
        logger.error(f"Erro ao carregar arquivo de configuração: {e}")
        raise

def load_dataset(processed_path: str) -> pd.DataFrame:
    """Lê o dataset em formato Parquet e retorna o DataFrame."""
    try:
        df = pd.read_parquet(processed_path)
        logger.info(f"Dataset carregado com {df.shape[0]} linhas e {df.shape[1]} colunas.")
        return df
    except Exception as e:
        logger.error(f"Erro ao ler o arquivo Parquet: {e}")
        raise

#############################
# Funções de Pré-processamento e Feature Engineering
#############################
def preprocess_data(df: pd.DataFrame,
                    date_col: str = 'data_hora_de_inicio_da_viagem',
                    metric_col: str = 'tarifa') -> pd.DataFrame:
    """
    Converte a coluna de data/hora para datetime, agrega os dados por data 
    (calculando a média da métrica) e gera features adicionais.
    
    Retorna um DataFrame com índice de data e as colunas:
      - metric: valor agregado (média)
      - rolling_mean: média móvel (janela de 7 dias)
      - rolling_std: desvio padrão móvel (janela de 7 dias)
      - trend: tendência extraída por STL
      - seasonal: componente sazonal extraído por STL
      - quantile_25: 25º percentil em janela de 7 dias
      - quantile_75: 75º percentil em janela de 7 dias
    """
    try:
        df[date_col] = pd.to_datetime(df[date_col])
    except Exception as e:
        logger.error(f"Erro na conversão da coluna de data: {e}")
        raise

    # Agregação diária
    df['data'] = df[date_col].dt.date
    serie = df.groupby('data')[metric_col].mean().sort_index()
    serie = serie.rename("metric")
    
    # Converter o índice para datetime (para operações de rolling e STL)
    serie.index = pd.to_datetime(serie.index)

    # Criação de features adicionais: rolling mean e rolling std com janela de 7 dias
    df_features = pd.DataFrame(serie)
    base_window = 7
    df_features['rolling_mean'] = df_features['metric'].rolling(window=base_window, min_periods=1).mean()
    df_features['rolling_std'] = df_features['metric'].rolling(window=base_window, min_periods=1).std().fillna(0)

    # Decomposição STL para extrair tendência e sazonalidade (usando período 7)
    try:
        stl = STL(df_features['metric'], period=7)
        res = stl.fit()
        df_features['trend'] = res.trend
        df_features['seasonal'] = res.seasonal
        logger.info("Decomposição STL concluída: trend e seasonal extraídos.")
    except Exception as e:
        logger.error(f"Erro na decomposição STL: {e}")
        raise

    # Exemplo de outras features: quantis
    df_features['quantile_25'] = df_features['metric'].rolling(window=base_window, min_periods=1).quantile(0.25)
    df_features['quantile_75'] = df_features['metric'].rolling(window=base_window, min_periods=1).quantile(0.75)

    logger.info("Pré-processamento e feature engineering concluídos.")
    return df_features

def segment_time_series(df_features: pd.DataFrame,
                        features: list = ['metric', 'rolling_mean', 'rolling_std', 'trend', 'seasonal', 'quantile_25', 'quantile_75'],
                        window_size: int = 30,
                        step: int = 15) -> np.ndarray:
    """
    Segmenta a série temporal em janelas de tamanho fixo (com possível sobreposição).
    
    Args:
        df_features (pd.DataFrame): DataFrame com índice de datas e features.
        features (list): Lista dos nomes das colunas a serem segmentadas.
        window_size (int): Tamanho da janela (em dias) para segmentação.
        step (int): Passo de deslocamento entre as janelas.
    
    Returns:
        np.ndarray: Array 3D com shape (n_segments, window_size, n_features).
    """
    data_array = df_features[features].values
    n_total = data_array.shape[0]
    n_features = data_array.shape[1]
    segments = []
    # Cria janelas sobrepostas: incrementa de acordo com o step
    for i in range(0, n_total - window_size + 1, step):
        segments.append(data_array[i:i+window_size, :])
    segments = np.array(segments)
    logger.info(f"Segmentação concluída: {segments.shape[0]} segmentos formados com {window_size} pontos cada e {n_features} features (step={step}).")
    return segments

def normalize_segments(segments: np.ndarray) -> (np.ndarray, StandardScaler):
    """
    Aplica normalização (StandardScaler) aos segmentos.
    Reestrutura os dados para aplicação do scaler e depois retorna os segmentos normalizados.
    """
    n_segments, window_size, n_features = segments.shape
    reshaped = segments.reshape(-1, n_features)
    scaler = StandardScaler()
    normalized = scaler.fit_transform(reshaped)
    normalized_segments = normalized.reshape(n_segments, window_size, n_features)
    logger.info("Normalização dos segmentos concluída.")
    return normalized_segments, scaler

#############################
# Funções de Clusterização e Otimização com Optuna
#############################
def objective(trial, segments_norm, df_features_global, feature_list_global):
    """
    Função objetivo para otimização dos hiperparâmetros com Optuna.
    Ajusta o número de clusters, o tamanho da janela e o passo, retornando o Silhouette Score.
    """
    n_clusters = trial.suggest_int('n_clusters', 2, 6)
    window_size = trial.suggest_int('window_size', 20, 40)
    step = trial.suggest_int('step', 10, window_size)
    
    # Re-segmenta os dados com os novos parâmetros
    segments = segment_time_series(df_features_global, features=feature_list_global, window_size=window_size, step=step)
    segments_norm_trial, _ = normalize_segments(segments)
    flattened = segments_norm_trial.reshape(segments_norm_trial.shape[0], -1)
    
    model = TimeSeriesKMeans(n_clusters=n_clusters, metric="dtw", max_iter=50, random_state=42)
    labels = model.fit_predict(segments_norm_trial)
    score = silhouette_score(flattened, labels)
    return score

def optimize_hyperparameters(segments_norm, df_features_global, feature_list_global):
    """
    Otimiza os hiperparâmetros usando Optuna para maximizar o Silhouette Score.
    """
    study = optuna.create_study(direction="maximize")
    study.optimize(lambda trial: objective(trial, segments_norm, df_features_global, feature_list_global), n_trials=20)
    best_params = study.best_params
    best_value = study.best_value
    logger.info(f"Otimização concluída. Melhores parâmetros: {best_params} com Silhouette Score = {best_value:.4f}")
    return best_params, best_value

def plot_cluster_centroids(model, window_size: int, n_features: int):
    """
    Plota os centróides dos clusters.
    Para séries multivariadas, plota cada feature em subplots.
    """
    for idx, centroid in enumerate(model.cluster_centers_):
        centroid = centroid.reshape(window_size, n_features)
        fig, axs = plt.subplots(n_features, 1, figsize=(8, 4 * n_features), sharex=True)
        if n_features == 1:
            axs = [axs]
        for i in range(n_features):
            axs[i].plot(centroid[:, i], marker='o', label=f'Feature: {i}')
            axs[i].set_title(f'Centróide do Cluster {idx} - Feature {i}')
            axs[i].set_xlabel('Tempo (dias)')
            axs[i].set_ylabel('Valor')
            axs[i].legend()
            axs[i].grid(True)
        plt.tight_layout()
        plt.show()

#############################
# Função para Modelos Alternativos de Clusterização
#############################
def run_alternative_models(segments_norm):
    """
    Executa modelos alternativos de clusterização (K-Shape e DBSCAN) e retorna suas métricas.
    """
    results = {}

    # Flatten os segmentos para métodos que não suportam diretamente séries temporais
    flattened = segments_norm.reshape(segments_norm.shape[0], -1)

    # Modelo 1: K-Shape (mantém o formato da série)
    try:
        kshape = KShape(n_clusters=2, max_iter=50, random_state=42)
        labels_kshape = kshape.fit_predict(segments_norm)
        score_kshape = silhouette_score(segments_norm.reshape(segments_norm.shape[0], -1), labels_kshape)
        results['KShape'] = {'n_clusters': 2, 'silhouette_score': score_kshape}
        logger.info(f"K-Shape: n_clusters=2, Silhouette Score = {score_kshape:.4f}")
    except Exception as e:
        logger.error(f"Erro no modelo K-Shape: {e}")

    # Modelo 2: DBSCAN aplicado no flattened segments
    try:
        # Ajuste os parâmetros de DBSCAN conforme necessário
        dbscan = DBSCAN(eps=0.5, min_samples=3, metric='euclidean')
        labels_dbscan = dbscan.fit_predict(flattened)
        # Verifica se mais de um cluster foi formado
        if len(set(labels_dbscan)) > 1 and -1 not in set(labels_dbscan):
            score_dbscan = silhouette_score(flattened, labels_dbscan)
            results['DBSCAN'] = {'eps': 0.5, 'min_samples': 3, 'silhouette_score': score_dbscan}
            logger.info(f"DBSCAN: Silhouette Score = {score_dbscan:.4f}")
        else:
            results['DBSCAN'] = {'eps': 0.5, 'min_samples': 3, 'silhouette_score': None}
            logger.info("DBSCAN: Não foi possível formar clusters válidos para cálculo do Silhouette Score.")
    except Exception as e:
        logger.error(f"Erro no modelo DBSCAN: {e}")
    
    return results

#############################
# Pipeline Principal com MLOps (MLflow)
#############################
def main():
    global df_features_global, feature_list_global  # para acesso na função objective
    # Caminhos (ajuste conforme necessário)
    config_file = r'D:\Github\data-science\projetos\logistica_transporte\3_analise_desempenho_viagens_agrupadas_clusterização_series_temporais\config\config.yaml'
    
    # Carregar configuração
    config = load_config(config_file)
    processed_data_path = config.get('data', {}).get('processed')
    if not processed_data_path:
        raise ValueError("Caminho para os dados processados não encontrado na configuração.")
    
    # Carregar dataset
    df = load_dataset(processed_data_path)
    logger.info("Dataset inicial:")
    logger.info(df.dtypes)
    
    # Pré-processamento e Feature Engineering (incluindo STL para trend e seasonal e novos quantis)
    df_features = preprocess_data(df, date_col='data_hora_de_inicio_da_viagem', metric_col='tarifa')
    logger.info(f"Exemplo de features calculadas:\n{df_features.head(10)}")
    
    # Atualiza a lista de features para incluir as novas extraídas
    feature_list = ['metric', 'rolling_mean', 'rolling_std', 'trend', 'seasonal', 'quantile_25', 'quantile_75']
    feature_list_global = feature_list  # variável global para otimização

    # Segmentação da série temporal com features multivariadas e janela de 30 dias com step de 15 (valores default)
    window_size_default = 30  
    step_default = 15         
    segments = segment_time_series(df_features, features=feature_list, window_size=window_size_default, step=step_default)
    
    # Normalização dos segmentos
    segments_norm, scaler = normalize_segments(segments)
    
    # Otimização Avançada com Optuna para ajustar hiperparâmetros
    df_features_global = df_features.copy()  # para acesso na função objective
    best_params, best_score = optimize_hyperparameters(segments_norm, df_features_global, feature_list_global)
    
    # Utiliza os melhores parâmetros encontrados para resegmentar e clusterizar os dados
    optimized_window = best_params.get('window_size', window_size_default)
    optimized_step = best_params.get('step', step_default)
    optimized_n_clusters = best_params.get('n_clusters', 2)
    
    # Re-segmentação com os parâmetros otimizados
    segments_opt = segment_time_series(df_features, features=feature_list, window_size=optimized_window, step=optimized_step)
    segments_norm_opt, _ = normalize_segments(segments_opt)
    flattened_opt = segments_norm_opt.reshape(segments_norm_opt.shape[0], -1)
    
    # Clusterização final com os melhores parâmetros usando TimeSeriesKMeans
    best_model = TimeSeriesKMeans(n_clusters=optimized_n_clusters, metric="dtw", max_iter=50, random_state=42)
    best_labels = best_model.fit_predict(segments_norm_opt)
    
    # Executa modelos alternativos de clusterização e coleta métricas
    alt_results = run_alternative_models(segments_norm_opt)
    
    # Registro dos resultados e parâmetros com MLflow
    mlflow.set_experiment("Clusterizacao_Series_Temporais")
    with mlflow.start_run():
        mlflow.log_param("feature_list", feature_list)
        mlflow.log_param("window_size_default", window_size_default)
        mlflow.log_param("step_default", step_default)
        mlflow.log_params(best_params)
        mlflow.log_metric("best_silhouette_score", best_score)
        mlflow.log_param("model_used_n_clusters", optimized_n_clusters)
        mlflow.log_param("alternative_models", alt_results)
        
        # Salva o modelo (opcional – para deploy)
        model_path = os.path.join(config.get('models', {}).get('directory', '.'), 'timeseries_kmeans_model.pkl')
        # Exemplo: salvar com joblib (descomente as linhas a seguir, se necessário)
        # import joblib
        # joblib.dump(best_model, model_path)
        mlflow.log_param("model_saved_path", model_path)
        
        # Plot dos centróides (para séries multivariadas)
        n_features = len(feature_list)
        plot_cluster_centroids(best_model, optimized_window, n_features)
    
    logger.info("Pipeline concluído com sucesso.")

if __name__ == '__main__':
    main()
