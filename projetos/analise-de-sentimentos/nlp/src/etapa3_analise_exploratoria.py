import dask.dataframe as dd
import matplotlib.pyplot as plt
import seaborn as sns
import logging
import yaml
from wordcloud import WordCloud
from PIL import Image

# Carregar arquivo de configuração
config_path = 'D:\\Github\\data-science\\projetos\\analise-de-sentimentos\\nlp\\config\\config.yaml'
with open(config_path, 'r') as file:
    config = yaml.safe_load(file)

# Configuração de Logs
logging.basicConfig(level=config['logs']['level'],
                    format=config['logs']['format'],
                    handlers=[logging.StreamHandler(),
                              logging.FileHandler(config['directories']['logs'] + 'analise_exploratoria.log')])
logger = logging.getLogger(__name__)

def analise_estatistica(df):
    """Descreve distribuições de frequências e calcula estatísticas descritivas"""
    describe_df = df.describe().compute()
    print(describe_df)
    return describe_df

def visualizacoes(df):
    """Cria visualizações como gráficos de dispersão, histogramas e box plots"""
    df_pd = df.compute()

    # Histograma de palavras
    df_pd['tweet_length'] = df_pd['tweet'].apply(len)
    plt.figure(figsize=(10, 6))
    sns.histplot(df_pd['tweet_length'], bins=50)
    plt.title("Distribuição do Comprimento dos Tweets")
    plt.xlabel("Comprimento do Tweet")
    plt.ylabel("Frequência")
    plt.savefig(config['directories']['figures'] + 'histograma_comprimento_tweets.png')

    # Nuvem de palavras
    wordcloud = WordCloud(width=800, height=400, max_words=200).generate(' '.join(df_pd['tweet']))
    plt.figure(figsize=(10, 6))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title("Nuvem de Palavras dos Tweets")
    plt.savefig(config['directories']['figures'] + 'nuvem_palavras_tweets.png')

def analise_correlacao(df):
    """Analisa correlações entre variáveis"""
    df_pd = df.compute()
    correlation_matrix = df_pd.corr()
    plt.figure(figsize=(10, 6))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
    plt.title("Matriz de Correlação")
    plt.savefig(config['directories']['figures'] + 'matriz_correlacao.png')

def analise_exploratoria():
    # Carregar o dataset pré-processado da etapa anterior
    processed_data_path = config['directories']['processed_data'] + config['files']['processed_dataset']
    logger.info("Carregando dataset pré-processado do caminho: %s", processed_data_path)

    df = dd.read_parquet(processed_data_path)
    logger.info("Dataset carregado com sucesso!")

    # Realizar análises
    describe_df = analise_estatistica(df)
    logger.info("Análise estatística realizada com sucesso!")

    visualizacoes(df)
    logger.info("Visualizações criadas com sucesso!")

    analise_correlacao(df)
    logger.info("Análise de correlação realizada com sucesso!")

    # Salvar o dataset analisado diretamente nos arquivos CSV e Parquet
    processed_data_parquet = config['directories']['processed_data'] + 'etapa3_analise_exploratoria.parquet'
    processed_data_csv = config['directories']['processed_data'] + 'etapa3_analise_exploratoria.csv'
    df.compute().to_parquet(processed_data_parquet, engine='pyarrow', index=False)
    df.compute().to_csv(processed_data_csv, index=False)
    logger.info("Dataset analisado salvo em %s e %s", processed_data_parquet, processed_data_csv)

    # Atualizar o arquivo de configuração
    config['files']['processed_dataset'] = 'etapa3_analise_exploratoria.parquet'
    with open(config_path, 'w') as file:
        yaml.dump(config, file)
    logger.info("Arquivo de configuração atualizado com sucesso.")

if __name__ == "__main__":
    analise_exploratoria()
