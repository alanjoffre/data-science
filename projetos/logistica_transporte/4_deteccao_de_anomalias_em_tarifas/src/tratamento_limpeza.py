#!/usr/bin/env python
import os
import time
import math
import yaml
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from datetime import datetime
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import logging

# Definir diretórios de logs e figures (gráficos)
logs_dir = r"D:\Github\data-science\projetos\logistica_transporte\4_deteccao_de_anomalias_em_tarifas\logs"
figures_dir = r"D:\Github\data-science\projetos\logistica_transporte\4_deteccao_de_anomalias_em_tarifas\reports\figures"
os.makedirs(logs_dir, exist_ok=True)
os.makedirs(figures_dir, exist_ok=True)

# Configurar logging para salvar mensagens no arquivo pipeline.log
logging.basicConfig(
    level=logging.INFO,
    filename=os.path.join(logs_dir, "pipeline.log"),
    filemode="w",
    format="%(asctime)s - %(levelname)s - %(message)s"
)

def load_config(config_path):
    """Carrega a configuração a partir de um arquivo YAML."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    logging.info("Configuração carregada do arquivo YAML.")
    return config

def load_dataset(path):
    """Importa o dataset a partir do arquivo Parquet."""
    df = pd.read_parquet(path)
    logging.info(f"Dataset carregado de: {path}")
    return df

def print_dataset_info(df):
    """Exibe informações básicas do dataset e as loga."""
    num_linhas, num_colunas = df.shape
    info = f"Número de linhas: {num_linhas}\nNúmero de colunas: {num_colunas}\n" + \
           "Colunas e tipos de dados:\n" + str(df.dtypes)
    print(info)
    logging.info(info)
    logging.info("-" * 50)

def verify_and_clean_data(df):
    """Verifica e corrige duplicados, missing values (com imputação),
       valores fora dos limites e outliers via IQR (winsorização),
       excluindo as colunas de setor do tratamento numérico.
       Retorna o dataframe limpo e um dicionário com as estatísticas."""
    logging.info("=== Início da Verificação e Limpeza de Dados ===")
    original_rows = df.shape[0]
    
    # 1. Duplicados
    n_duplicates = df.duplicated().sum()
    logging.info(f"Registros duplicados encontrados: {n_duplicates} ({(n_duplicates/original_rows)*100:.2f}%)")
    df = df.drop_duplicates()
    duplicates_removed = n_duplicates
    logging.info(f"Registros duplicados removidos: {duplicates_removed}")
    print("-" * 50)
    
    # 2. Tratamento de Missing Values (Imputação)
    missing_before = df.isnull().sum().sum()
    if missing_before > 0:
        logging.info("Valores faltantes antes da imputação por coluna:")
        logging.info(df.isnull().sum()[df.isnull().sum() > 0])
        for col in df.columns:
            if df[col].isnull().sum() > 0:
                if pd.api.types.is_numeric_dtype(df[col]):
                    df[col].fillna(df[col].median(), inplace=True)
                else:
                    df[col].fillna(df[col].mode()[0], inplace=True)
    else:
        logging.info("Não foram encontrados dados faltantes.")
    missing_after = df.isnull().sum().sum()
    missing_imputed = missing_before - missing_after
    logging.info(f"Total de valores faltantes imputados: {missing_imputed} (100.00% resolvidos)")
    print("-" * 50)
    
    # Converter as colunas de setor para categoria para evitar tratamento numérico
    for col in ['setor_censitario_de_partida', 'setor_censitario_de_destino']:
        if col in df.columns:
            df[col] = df[col].astype('category')
            logging.info(f"Coluna {col} convertida para categoria.")
    print("-" * 50)
    
    # 3. Valores fora dos limites (apenas para variáveis numéricas)
    limites = {
        'duracao_segundos_da_viagem': (0, 10000),
        'distancia_milhas_da_viagem': (0, 500),
        # As colunas de setor foram convertidas para categoria, serão ignoradas.
        'area_comunitaria_de_partida': (0, 10000),
        'area_comunitaria_de_destino': (0, 10000),
        'tarifa': (0, 200),
        'gorjeta': (0, 50),
        'cobrancas_adicionais': (0, 50),
        'total_da_viagem': (0, 300),
        'viagens_compartilhadas': (0, 5),
        'latitude_centro_de_partida': (-90, 90),
        'longitude_centro_de_partida': (-180, 180),
        'latitude_centro_de_destino': (-90, 90),
        'longitude_centro_de_destino': (-180, 180),
        'tarifa_por_milha': (0, 10),
        'porcentagem_gorjeta': (0, 100),
        'velocidade_media': (0, 150),
        'razao_cobrancas': (0, 10)
    }
    
    total_corrections_limits = 0
    logging.info("Verificação de valores fora dos limites:")
    for col, (lim_inf, lim_sup) in limites.items():
        if col in df.columns:
            if col in ['setor_censitario_de_partida', 'setor_censitario_de_destino']:
                logging.info(f" - {col}: tratamento numérico não aplicado (variável categórica)")
                continue
            n_out = df[(df[col] < lim_inf) | (df[col] > lim_sup)].shape[0]
            logging.info(f" - {col}: {n_out} registros fora do limite [{lim_inf}, {lim_sup}]")
            total_corrections_limits += n_out
            df[col] = df[col].clip(lower=lim_inf, upper=lim_sup)
    logging.info(f"Total de correções aplicadas (valores fora dos limites): {total_corrections_limits} ({(total_corrections_limits/original_rows)*100:.2f}%)")
    print("-" * 50)
    
    # 4. Outliers via IQR (winsorização) para variáveis numéricas (exceto as de setor)
    outlier_exclusions = ['setor_censitario_de_partida', 'setor_censitario_de_destino']
    total_outlier_corrections = 0
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    logging.info("Detecção e correção de outliers (método IQR):")
    for col in numeric_cols:
        if col in outlier_exclusions:
            logging.info(f" - {col}: winsorização não aplicada (excluída)")
            continue
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_fence = Q1 - 1.5 * IQR
        upper_fence = Q3 + 1.5 * IQR
        n_outliers = df[(df[col] < lower_fence) | (df[col] > upper_fence)].shape[0]
        logging.info(f" - {col}: {n_outliers} outliers detectados e corrigidos")
        total_outlier_corrections += n_outliers
        df[col] = df[col].clip(lower=lower_fence, upper=upper_fence)
    logging.info(f"Total de correções aplicadas (outliers via IQR): {total_outlier_corrections} ({(total_outlier_corrections/original_rows)*100:.2f}%)")
    print("-" * 50)
    
    final_rows = df.shape[0]
    logging.info(f"Número final de linhas: {final_rows} ({(final_rows/original_rows)*100:.2f}% do total original)")
    logging.info("=== Fim da Verificação e Limpeza ===\n")
    
    stats = {
        'original_rows': original_rows,
        'duplicates_removed': n_duplicates,
        'missing_imputed': missing_imputed,
        'total_corrections_limits': total_corrections_limits,
        'total_outlier_corrections': total_outlier_corrections,
        'final_rows': final_rows
    }
    return df, stats

def preprocess_data(df):
    """Realiza pré-processamento, conversão de datas, conversão de variáveis de setor para categóricas e engenharia de features."""
    # Converter colunas de data para datetime
    date_cols = ['data_hora_de_inicio_da_viagem', 'data_hora_de_termino_da_viagem']
    for col in date_cols:
        df[col] = pd.to_datetime(df[col], errors='coerce', format='%m/%d/%Y %I:%M:%S %p')
    
    # Converter as colunas de setor para variáveis categóricas (caso ainda não estejam)
    for col in ['setor_censitario_de_partida', 'setor_censitario_de_destino']:
        if col in df.columns and not pd.api.types.is_categorical_dtype(df[col]):
            df[col] = df[col].astype('category')
    
    # Feature Engineering: criar duração em minutos (convertendo de segundos)
    df['duracao_minutos'] = df['duracao_segundos_da_viagem'] / 60.0
    
    # Extração de features temporais
    df['inicio_hora'] = df['data_hora_de_inicio_da_viagem'].dt.hour
    df['inicio_dia'] = df['data_hora_de_inicio_da_viagem'].dt.day
    df['inicio_mes'] = df['data_hora_de_inicio_da_viagem'].dt.month
    df['inicio_dia_semana'] = df['data_hora_de_inicio_da_viagem'].dt.weekday

    return df

def scale_features(df, feature_cols):
    """Realiza o escalonamento das features numéricas utilizando StandardScaler."""
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df[feature_cols])
    return X_scaled, scaler

def train_anomaly_model(X):
    """Treina um modelo de detecção de anomalias usando IsolationForest."""
    model = IsolationForest(n_estimators=100, contamination=0.05, random_state=42)
    model.fit(X)
    return model

def explore_setor_columns(df, output_file="exploracao_setor.txt"):
    """Gera uma análise exploratória das colunas de setor, salvando as estatísticas e os gráficos."""
    columns = ['setor_censitario_de_partida', 'setor_censitario_de_destino']
    with open(output_file, 'w') as f:
        for col in columns:
            f.write(f"===== Análise de {col} =====\n")
            print(f"===== Análise de {col} =====")
            
            # Estatísticas descritivas
            desc = df[col].describe()
            f.write("Estatísticas descritivas:\n")
            f.write(desc.to_string())
            f.write("\n\n")
            print("Estatísticas descritivas:")
            print(desc)
            print("\n")
            
            # Frequência de valores únicos
            counts = df[col].value_counts().sort_index()
            f.write("Contagem de valores únicos (top 20):\n")
            f.write(counts.head(20).to_string())
            f.write("\n...\n\n")
            print("Contagem de valores únicos:")
            print(counts.head(20))
            print("...\n")
            
            # Histograma
            plt.figure(figsize=(8, 4))
            sns.histplot(df[col], bins=50, kde=False, color='skyblue')
            plt.title(f"Histograma de {col}")
            plt.xlabel(col)
            plt.ylabel("Frequência")
            hist_path = os.path.join(figures_dir, f"histograma_{col}.png")
            plt.savefig(hist_path)
            plt.close()
            f.write(f"Histograma salvo em: {hist_path}\n")
            print(f"Histograma salvo em: {hist_path}")
            
            # Se a variável for numérica, utiliza boxplot; se for categórica, countplot
            if pd.api.types.is_numeric_dtype(df[col]):
                plt.figure(figsize=(6, 4))
                sns.boxplot(x=df[col], color='lightgreen')
                plt.title(f"Boxplot de {col}")
                plt.xlabel(col)
                plot_path = os.path.join(figures_dir, f"boxplot_{col}.png")
            else:
                plt.figure(figsize=(8, 4))
                sns.countplot(x=df[col], color='lightgreen')
                plt.title(f"Countplot de {col}")
                plt.xlabel(col)
                plot_path = os.path.join(figures_dir, f"countplot_{col}.png")
            plt.savefig(plot_path)
            plt.close()
            f.write(f"Gráfico salvo em: {plot_path}\n")
            print(f"Gráfico salvo em: {plot_path}")
            
            f.write("\n" + "="*50 + "\n\n")
            print("\n" + "="*50 + "\n")

def main():
    start_time = time.time()
    
    # 1. Carregar a configuração
    config_path = os.path.join('D:/Github/data-science/projetos/logistica_transporte/4_deteccao_de_anomalias_em_tarifas/config', 'config.yaml')
    config = load_config(config_path)
    
    # 2. Definir os caminhos a partir do arquivo de configuração
    data_path = config['data']['raw']
    model_dir = config['models']['directory']
    os.makedirs(model_dir, exist_ok=True)
    
    processed_parquet_path = config['data']['processed_parquet']
    processed_csv_path = config['data']['processed_csv']
    
    # 3. Importar o dataset raw
    df = load_dataset(data_path)
    print("=== Informações Iniciais do Dataset ===")
    print_dataset_info(df)
    
    # (Opcional) Análise exploratória para as colunas de setor antes da conversão
    explore_setor_columns(df)
    
    # 4. Verificar e limpar dados: duplicados, missing (com imputação), limites e outliers
    df, stats = verify_and_clean_data(df)
    print("=== Informações Após Limpeza ===")
    print_dataset_info(df)
    
    print("=== Resumo das Operações de Limpeza ===")
    print(f"Registros eliminados por duplicidade: {stats['duplicates_removed']} (0.00% caso 0)")
    print(f"Valores faltantes imputados: {stats['missing_imputed']} (100% resolvidos)")
    print(f"Correções aplicadas (valores fora dos limites): {stats['total_corrections_limits']} ({(stats['total_corrections_limits']/stats['original_rows'])*100:.2f}%)")
    print(f"Correções aplicadas (outliers via IQR): {stats['total_outlier_corrections']} ({(stats['total_outlier_corrections']/stats['original_rows'])*100:.2f}%)")
    print(f"Número final de linhas: {stats['final_rows']} ({(stats['final_rows']/stats['original_rows'])*100:.2f}% do total original)")
    print("=" * 50)
    
    # 5. Pré-processamento e Engenharia de Features (incluindo conversão de setor para categórica)
    df = preprocess_data(df)
    print("=== Informações Após Pré-processamento ===")
    print_dataset_info(df)
    
    # (Opcional) Análise exploratória para as colunas de setor após conversão
    explore_setor_columns(df, output_file="exploracao_setor_pos.txt")
    
    # 6. Selecionar features para a detecção de anomalias
    feature_cols = ['tarifa', 'gorjeta', 'total_da_viagem', 'duracao_minutos',
                    'distancia_milhas_da_viagem', 'tarifa_por_milha']
    feature_cols = [col for col in feature_cols if col in df.columns]
    
    # 7. Escalonamento das features
    X, scaler = scale_features(df, feature_cols)
    
    # 8. Treinar o modelo de detecção de anomalias
    model = train_anomaly_model(X)
    
    # 9. Aplicar o modelo para identificar anomalias
    df['anomaly'] = model.predict(X)
    n_anomalias = (df['anomaly'] == -1).sum()
    print(f"\nTotal de anomalias detectadas: {n_anomalias}")
    
    # 10. Salvar os artefatos do modelo e do escalonador
    joblib.dump(model, os.path.join(model_dir, 'final_model.joblib'))
    joblib.dump(scaler, os.path.join(model_dir, 'scaler.joblib'))
    print(f"Modelo e scaler salvos em: {model_dir}")
    
    # 11. Salvar o dataset processado em Parquet e CSV
    df.to_parquet(processed_parquet_path, index=False)
    df.to_csv(processed_csv_path, index=False)
    print(f"Dataset processado salvo em:\n  {processed_parquet_path}\n  {processed_csv_path}")
    
    # Medir e exibir o tempo total de processamento no formato HHhr MMmin SSseg
    total_time = time.time() - start_time
    hours = math.floor(total_time / 3600)
    minutes = math.floor((total_time % 3600) / 60)
    seconds = total_time % 60
    time_formatted = f"{hours:02d}hr {minutes:02d}min {seconds:05.2f}seg"
    print(f"\nTempo total de processamento: {time_formatted}")
    logging.info(f"Tempo total de processamento: {time_formatted}")

if __name__ == "__main__":
    main()
