import dask.dataframe as dd
import yaml
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Caminho para o arquivo vetorizado
dataset_path = 'D:\\Github\\data-science\\projetos\\analise-de-sentimentos\\nlp\\data\\processed\\etapa2_6_vetorizacao_texto.parquet'

# Carregar o dataset vetorizado usando Dask
df = dd.read_parquet(dataset_path)

# Calcular Estatísticas Descritivas
estatisticas_descritivas = df.describe().compute()

# Salvar Estatísticas Descritivas em CSV
estatisticas_descritivas.to_csv('D:\\Github\\data-science\\projetos\\analise-de-sentimentos\\nlp\\data\\processed\\etapa3_1_analise_estatistica.csv')

# Salvar Estatísticas Descritivas em Parquet
estatisticas_descritivas.to_parquet('D:\\Github\\data-science\\projetos\\analise-de-sentimentos\\nlp\\data\\processed\\etapa3_1_analise_estatistica.parquet')

# Função para salvar gráficos
def salvar_grafico(fig, nome_arquivo):
    figures_path = 'D:\\Github\\data-science\\projetos\\analise-de-sentimentos\\nlp\\reports\\figures\\'
    fig.savefig(os.path.join(figures_path, nome_arquivo))
    plt.close(fig)

# Análise de Colunas Específicas
def analise_colunas(df):
    # Histograma da coluna sentimento_codificado
    fig, ax = plt.subplots()
    df['sentimento_codificado'].compute().hist(ax=ax, bins=20)
    ax.set_title('Distribuição dos Sentimentos Codificados')
    salvar_grafico(fig, 'sentimento_codificado_histograma.png')

    # Gráfico de Barras da coluna sentimento
    fig, ax = plt.subplots()
    df['sentimento'].compute().value_counts().plot(kind='bar', ax=ax)
    ax.set_title('Contagem de Sentimentos')
    salvar_grafico(fig, 'sentimento_barras.png')

    # Boxplot da vetorização dos tweets (exemplo usando a primeira dimensão do vetor)
    vetorizacao_df = df['vetorizacao_tweet'].compute()
    vetorizacao_array = vetorizacao_df.apply(lambda x: eval(x)[0])  # Exemplo: primeira dimensão do vetor
    fig, ax = plt.subplots()
    sns.boxplot(x=vetorizacao_array, ax=ax)
    ax.set_title('Boxplot da Primeira Dimensão da Vetorização dos Tweets')
    salvar_grafico(fig, 'vetorizacao_tweet_boxplot.png')

    # Contagem de linhas por sentimento_vader
    contagem_vader = df['sentimento_vader'].compute().value_counts()
    print("Contagem de linhas por sentimento_vader:")
    print(contagem_vader)

    # Contagem de linhas por sentimento_bert
    contagem_bert = df['sentimento_bert'].compute().value_counts()
    print("\nContagem de linhas por sentimento_bert:")
    print(contagem_bert)

    # Contagem de linhas por sentimento_codificado
    contagem_codificado = df['sentimento_codificado'].compute().value_counts()
    print("\nContagem de linhas por sentimento_codificado:")
    print(contagem_codificado)

analise_colunas(df)

# Atualizar o arquivo de configuração
config = {
    'directories': {
        'processed_data': 'D:\\Github\\data-science\\projetos\\analise-de-sentimentos\\nlp\\data\\processed\\',
        'figures': 'D:\\Github\\data-science\\projetos\\analise-de-sentimentos\\nlp\\reports\\figures\\',
        'logs': 'D:\\Github\\data-science\\projetos\\analise-de-sentimentos\\nlp\\logs\\'
    },
    'files': {
        'log_file': 'etapa2_3_analise_estatistica.log',
        'processed_dataset': 'etapa2_6_vetorizacao_texto.parquet',
        'processed_dataset_csv': 'etapa2_6_vetorizacao_texto.csv',
        'raw_dataset': 'asentimentos.parquet'
    }
}

with open('D:\\Github\\data-science\\projetos\\analise-de-sentimentos\\nlp\\config\\config.yaml', 'w') as file:
    yaml.dump(config, file)

print("Análise Estatística e Visualização de Dados concluídas com sucesso.")
