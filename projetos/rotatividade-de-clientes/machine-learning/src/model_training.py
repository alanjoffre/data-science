import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
import yaml
import os
import logging
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc, confusion_matrix, ConfusionMatrixDisplay

# Configurar logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def carregar_configuracao(config_path='D:\\Github\\data-science\\projetos\\rotatividade-de-clientes\\machine-learning\\config\\config.yaml'):
    """
    Carregar configurações de um arquivo YAML.

    Parameters:
    config_path (str): O caminho para o arquivo de configuração YAML.

    Returns:
    dict: Um dicionário contendo as configurações carregadas do arquivo.

    Raises:
    FileNotFoundError: Se o arquivo de configuração não for encontrado.
    YAMLError: Se houver um erro ao ler o arquivo YAML.
    """
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

def validar_colunas(df, cols):
    """
    Valida se as colunas necessárias estão presentes no DataFrame.

    Parameters:
    df (pd.DataFrame): O DataFrame a ser validado.
    cols (list): Lista de colunas necessárias.

    Raises:
    ValueError: Se alguma coluna necessária estiver ausente no DataFrame.
    """
    missing_cols = [col for col in cols if col not in df.columns]
    if missing_cols:
        logger.error(f"Colunas ausentes no dataset: {missing_cols}")
        raise ValueError(f"Colunas ausentes: {missing_cols}")
    logger.info("Todas as colunas necessárias estão presentes.")

def carregar_dados(config):
    """
    Carregar dados a partir do caminho especificado no arquivo de configuração.

    Parameters:
    config (dict): Dicionário contendo as configurações do projeto, incluindo o caminho dos dados.

    Returns:
    pd.DataFrame: DataFrame contendo os dados carregados.

    Raises:
    Exception: Se houver um erro ao carregar os dados.
    """
    raw_data_path = config['data']['raw']
    try:
        df = pd.read_csv(raw_data_path)
        logger.info(f'Dataset bruto carregado de: {raw_data_path}')
        return df
    except Exception as e:
        logger.error(f"Erro ao carregar o dataset bruto: {e}")
        raise

def tratar_dados(df, num_cols, config):
    """
    Tratar valores ausentes e duplicados no dataset e salvar os dados tratados em um arquivo CSV.

    Parameters:
    df (pd.DataFrame): DataFrame contendo os dados brutos.
    num_cols (list): Lista de colunas numéricas que precisam ser tratadas.
    config (dict): Dicionário contendo as configurações do projeto, incluindo o caminho para salvar os dados tratados.

    Returns:
    pd.DataFrame: DataFrame com valores ausentes tratados e duplicados removidos.
    """
    # Tratar valores ausentes
    logger.info("Verificando dados ausentes...")
    missing_values = df.isnull().sum()
    logger.info(f"Valores ausentes por coluna:\n{missing_values[missing_values > 0]}")

    for col in num_cols:
        if col in df.columns:
            mean_value = df[col].mean()
            df[col] = df[col].fillna(mean_value)
            logger.info(f"Valores ausentes na coluna '{col}' substituídos pela média: {mean_value}.")
        else:
            logger.warning(f"Coluna '{col}' não encontrada no dataset.")

    # Verificar e remover duplicados
    logger.info("Verificando duplicados...")
    duplicates = df.duplicated().sum()
    logger.info(f"Número de duplicados encontrados: {duplicates}")
    if duplicates > 0:
        df = df.drop_duplicates()
        logger.info("Duplicados removidos.")
    
    # Salvar dados tratados em um arquivo CSV
    processed_data_path = config['data']['data_processed']
    df.to_csv(processed_data_path, index=False)
    logger.info(f"Dados tratados salvos em: {processed_data_path}")

    return df

def criar_preprocessador(num_cols):
    """
    Criar um pré-processador para colunas numéricas.

    Parameters:
    num_cols (list): Lista de colunas numéricas para normalização.

    Returns:
    ColumnTransformer: Um objeto ColumnTransformer configurado para normalização das colunas numéricas.
    """
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), num_cols)  # Normalização das colunas numéricas
        ]
    )
    return preprocessor

def aplicar_smote(X, y):
    """
    Aplicar SMOTE para balanceamento de classes.

    Parameters:
    X (pd.DataFrame): DataFrame com as características (features) dos dados.
    y (pd.Series): Série com as etiquetas (labels) dos dados.

    Returns:
    pd.DataFrame, pd.Series: DataFrame e Série balanceados.
    """
    logger.info("Aplicando SMOTE para balanceamento de classes...")
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X, y)
    logger.info("Balanceamento de classes concluído.")
    logger.info(f"Nova distribuição de classes:\n{pd.Series(y_resampled).value_counts()}")
    return X_resampled, y_resampled

def salvar_preprocessador(preprocessor, preprocessor_path):
    """
    Salvar o pré-processador.

    Parameters:
    preprocessor (ColumnTransformer): O pré-processador a ser salvo.
    preprocessor_path (str): O caminho onde o pré-processador será salvo.
    """
    try:
        joblib.dump(preprocessor, preprocessor_path)
        logger.info(f'Pré-processador salvo em: {preprocessor_path}')
    except Exception as e:
        logger.error(f"Erro ao salvar o pré-processador: {e}")
        raise

def obter_nomes_colunas(preprocessor, num_cols):
    """
    Obter os nomes das colunas originais após a aplicação do pré-processador.

    Parameters:
    preprocessor (ColumnTransformer): O pré-processador aplicado.
    num_cols (list): Lista de colunas numéricas originais.

    Returns:
    list: Lista de nomes de colunas originais.
    """
    return num_cols

def salvar_dataset_processado(X, y, processed_path, original_cols):
    """
    Salvar o dataset processado em um arquivo CSV e imprimir as primeiras 5 linhas no console.

    Parameters:
    X (pd.DataFrame): DataFrame contendo as características (features) dos dados.
    y (pd.Series): Série contendo as etiquetas (labels) dos dados.
    processed_path (str): Caminho onde o dataset processado será salvo.
    original_cols (list): Lista de nomes das colunas originais.

    Raises:
    Exception: Se houver um erro ao salvar o dataset processado.
    """
    df_resampled = pd.concat([pd.DataFrame(X, columns=original_cols), pd.Series(y, name='Exited')], axis=1)
    try:
        df_resampled.to_csv(processed_path, index=False)
        logger.info(f'Dataset processado salvo em: {processed_path}')
        print("\nPrimeiras 5 linhas do dataset processado:\n", df_resampled.head())
    except Exception as e:
        logger.error(f"Erro ao salvar o dataset processado: {e}")
        raise

def gerar_graficos(df, output_dir):
    """
    Gerar gráficos do dataset preprocessado e salvar no diretório especificado.

    Parameters:
    df (pd.DataFrame): DataFrame contendo os dados preprocessados.
    output_dir (str): Diretório onde os gráficos serão salvos.

    Raises:
    Exception: Se houver um erro ao salvar os gráficos.
    """
    logger.info("Gerando gráficos...")

    # Verificar se o diretório existe, se não, criar
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Histograma de Idades
    plt.figure(figsize=(10, 6))
    sns.histplot(df['Age'], kde=True)
    plt.title('Distribuição de Idades')
    plt.xlabel('Idade')
    plt.ylabel('Frequência')
    plt.savefig(os.path.join(output_dir, 'distribuicao_idades.png'))
    plt.close()

    # Box Plot de EstimatedSalary por Exited
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='Exited', y='EstimatedSalary', data=df)
    plt.title('Salário Estimado vs. Exited')
    plt.xlabel('Exited')
    plt.ylabel('Salário Estimado')
    plt.savefig(os.path.join(output_dir, 'salario_vs_exited.png'))
    plt.close()

    # Scatter Plot de Idade vs. Saldo
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='Age', y='Balance', hue='Exited', data=df)
    plt.title('Idade vs. Saldo')
    plt.xlabel('Idade')
    plt.ylabel('Saldo')
    plt.savefig(os.path.join(output_dir, 'scatter_idade_vs_saldo.png'))
    plt.close()

    # Gráfico de Linha de Saldo ao Longo dos Anos
    plt.figure(figsize=(10, 6))
    sns.lineplot(x='Tenure', y='Balance', hue='Exited', data=df)
    plt.title('Saldo ao Longo dos Anos de Permanência')
    plt.xlabel('Anos de Permanência')
    plt.ylabel('Saldo')
    plt.savefig(os.path.join(output_dir, 'line_saldo_anos.png'))
    plt.close()

    # Heatmap de Correlação
    plt.figure(figsize=(12, 10))
    correlation_matrix = df.corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
    plt.title('Mapa de Calor de Correlação')
    plt.savefig(os.path.join(output_dir, 'heatmap_correlacao.png'))
    plt.close()

    # Histograma de Distribuição do Score de Crédito
    plt.figure(figsize=(10, 6))
    sns.histplot(df['CreditScore'], kde=True)
    plt.title('Distribuição do Score de Crédito')
    plt.xlabel('Score de Crédito')
    plt.ylabel('Frequência')
    plt.savefig(os.path.join(output_dir, 'distribuicao_credit_score.png'))
    plt.close()

    # Pairplot das Variáveis com Hue para Exited
    plt.figure(figsize=(12, 10))
    sns.pairplot(df[['Balance', 'Age', 'CreditScore', 'Tenure', 'EstimatedSalary', 'Exited']], hue='Exited', diag_kind='kde')
    plt.savefig(os.path.join(output_dir, 'pairplot_com_hue.png'))
    plt.close()

    logger.info("Gráficos gerados e salvos com sucesso.")

def preprocessamento_dados():
    """
    Função principal para preprocessar dados.

    Este pipeline inclui as seguintes etapas:
    1. Carregar configurações do arquivo YAML.
    2. Carregar o dataset bruto.
    3. Validar a presença de colunas necessárias.
    4. Tratar valores ausentes e duplicados.
    5. Criar e aplicar o pré-processador para normalização das colunas numéricas.
    6. Aplicar SMOTE para balanceamento de classes.
    7. Salvar o pré-processador.
    8. Gerar gráficos detalhados do dataset preprocessado.
    9. Salvar o dataset processado em um arquivo CSV e imprimir as primeiras 5 linhas.

    Raises:
    Exception: Se houver um erro em qualquer etapa do processamento.
    """
    config = carregar_configuracao()
    
    # Carregar o dataset bruto
    df = carregar_dados(config)

    # Definir colunas numéricas
    num_cols = ['Balance', 'Age', 'CreditScore', 'Tenure', 'EstimatedSalary']
    required_cols = num_cols + ['Exited']

    # Validar colunas
    validar_colunas(df, required_cols)

    # Tratar dados
    df = tratar_dados(df, num_cols, config)

    # Criar e aplicar o pré-processador
    preprocessor = criar_preprocessador(num_cols)
    X = df.drop('Exited', axis=1)
    y = df['Exited']
    X_transformed = preprocessor.fit_transform(X)
    logger.info("Pré-processamento concluído.")

    # Obter nomes das colunas após o pré-processamento
    colunas_originais = obter_nomes_colunas(preprocessor, num_cols)

    # Aplicar SMOTE para balanceamento de classes
    X_resampled, y_resampled = aplicar_smote(X_transformed, y)

    # Salvar o pré-processador
    salvar_preprocessador(preprocessor, config['preprocessors']['path'])

    # Gerar gráficos com o dataset pré-processado
    df_resampled = pd.concat([pd.DataFrame(X_resampled, columns=colunas_originais), pd.Series(y_resampled, name='Exited')], axis=1)
    gerar_graficos(df_resampled, config['reports']['figures_dir'])

    # Salvar o dataset processado e imprimir as primeiras 5 linhas
    salvar_dataset_processado(X_resampled, y_resampled, config['data']['processed'], colunas_originais)

if __name__ == "__main__":
    preprocessamento_dados()