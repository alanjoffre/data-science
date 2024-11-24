import dask.dataframe as dd
import matplotlib.pyplot as plt
import seaborn as sns
import yaml
import os
import numpy as np

def gerar_visualizacoes(config_path):
    # Carregar arquivo de configuração
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)

    # Caminho para o arquivo vetorizado
    dataset_path = os.path.join(config['directories']['processed_data'], config['files']['processed_dataset'])

    # Carregar o dataset vetorizado usando Dask
    df = dd.read_parquet(dataset_path)

    # Caminho para salvar as figuras
    save_path = config['directories']['figures']
    print(f"Salvando figuras em: {save_path}")

    # Verificar se o diretório de figuras existe, caso contrário, criar
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # Função para salvar gráficos
    def salvar_grafico(fig, nome_arquivo):
        fig.savefig(os.path.join(save_path, nome_arquivo))
        plt.close(fig)
        print(f"Gráfico salvo: {nome_arquivo}")

    # Função para gerar histogramas
    def gerar_histograma(df, coluna):
        fig, ax = plt.subplots()
        df[coluna].compute().hist(ax=ax, bins=30)
        ax.set_title(f'Histograma de {coluna}')
        salvar_grafico(fig, f'histograma_{coluna}.png')

    # Função para gerar gráficos de barras
    def gerar_grafico_barras(df, coluna):
        fig, ax = plt.subplots()
        df[coluna].compute().value_counts().plot(kind='bar', ax=ax)
        ax.set_title(f'Gráfico de Barras de {coluna}')
        salvar_grafico(fig, f'bar_{coluna}.png')

    # Função para gerar box plots
    def gerar_boxplot(df, coluna):
        fig, ax = plt.subplots()
        sns.boxplot(x=df[coluna].compute(), ax=ax)
        ax.set_title(f'Boxplot de {coluna}')
        salvar_grafico(fig, f'boxplot_{coluna}.png')

    # Função para gerar gráficos de dispersão
    def gerar_grafico_dispersao(df, coluna_x, coluna_y):
        fig, ax = plt.subplots()
        sns.scatterplot(x=df[coluna_x].compute(), y=df[coluna_y].compute(), ax=ax)
        ax.set_title(f'Gráfico de Dispersão de {coluna_x} vs {coluna_y}')
        salvar_grafico(fig, f'scatter_{coluna_x}_vs_{coluna_y}.png')

    # Colunas para visualizações específicas
    colunas_histogramas = ['sentimento_codificado', 'sentimento_vader', 'sentimento_bert']
    colunas_barras = ['sentimento', 'username']
    colunas_boxplot = ['sentimento_codificado']

    # Gerar visualizações
    for coluna in colunas_histogramas:
        gerar_histograma(df, coluna)

    for coluna in colunas_barras:
        gerar_grafico_barras(df, coluna)

    for coluna in colunas_boxplot:
        gerar_boxplot(df, coluna)

    # Gráfico de dispersão entre duas colunas
    gerar_grafico_dispersao(df, 'sentimento_codificado', 'sentimento')

    # Análise da coluna 'vetorizacao_tweet' para boxplot da primeira dimensão do vetor
    vetorizacao_df = df['vetorizacao_tweet'].apply(lambda x: eval(x), meta=('vetorizacao_tweet', object)).compute()
    vetorizacao_array = vetorizacao_df.apply(lambda x: x[0] if isinstance(x, list) else np.nan).dropna()  # Exemplo: primeira dimensão do vetor
    fig, ax = plt.subplots()
    sns.boxplot(x=vetorizacao_array, ax=ax)
    ax.set_title('Boxplot da Primeira Dimensão da Vetorização dos Tweets')
    salvar_grafico(fig, 'boxplot_vetorizacao_tweet.png')

    print("Visualizações concluídas com sucesso.")

# Função para atualizar o arquivo de configuração
def atualizar_config(config_path, updates):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    config.update(updates)
    with open(config_path, 'w') as file:
        yaml.safe_dump(config, file)
    print("Arquivo config.yaml atualizado com sucesso.")

if __name__ == "__main__":
    config_path = 'D:\\Github\\data-science\\projetos\\analise-de-sentimentos\\nlp\\config\\config.yaml'
    gerar_visualizacoes(config_path)

    # Atualizações a serem feitas no config.yaml
    updates = {
    'directories': {
            'processed_data': 'D:\\Github\\data-science\\projetos\\analise-de-sentimentos\\nlp\\data\\processed\\',
            'figures': 'D:\\Github\\data-science\\projetos\\analise-de-sentimentos\\nlp\\reports\\figures\\',
            'logs': 'D:\\Github\\data-science\\projetos\\analise-de-sentimentos\\nlp\\logs\\'
    },
    'files': {
        'log_file': 'etapa3_2_visualizacoes.log',
        'processed_dataset': 'etapa2_6_vetorizacao_texto.parquet',
        'processed_dataset_csv': 'etapa2_6_vetorizacao_texto.csv',
        'raw_dataset': 'asentimentos.parquet'
        }
    }

    atualizar_config(config_path, updates)
