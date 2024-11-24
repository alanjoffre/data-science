import dask.dataframe as dd
import logging
import yaml
import os
import spacy
import pandas as pd
from nltk.stem import SnowballStemmer
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from transformers import pipeline
from sklearn.preprocessing import LabelEncoder
import pyarrow as pa
import pyarrow.parquet as pq

# Carregar arquivo de configuração
config_path = 'D:\\Github\\data-science\\projetos\\analise-de-sentimentos\\nlp\\config\\config.yaml'
with open(config_path, 'r') as file:
    config = yaml.safe_load(file)

# Configuração de Logs
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[logging.StreamHandler(),
                              logging.FileHandler(os.path.join(config['directories']['logs'], "etapa2_5_stemming_lemmatizacao.log"))])
logger = logging.getLogger(__name__)

def atualizar_config(config, chave, valor):
    config[chave] = valor
    with open(config_path, 'w') as file:
        yaml.safe_dump(config, file)
    logger.info("Arquivo config.yaml atualizado com sucesso.")

# Carregar modelo de linguagem spaCy
nlp = spacy.load("en_core_web_sm")

# Inicializar o stemmer do NLTK
stemmer = SnowballStemmer(language='english')

def aplicar_stemming(tokens):
    return [stemmer.stem(token) for token in tokens]

def aplicar_lemmatizacao(tokens):
    doc = nlp(" ".join(tokens))
    return [token.lemma_.strip() for token in doc if token.lemma_.strip()]

def remover_documentos_vazios(df):
    try:
        logger.info("Removendo documentos vazios ou contendo apenas stop words.")
        total_linhas = len(df)
        logger.info(f"Total de linhas antes da remoção: {total_linhas}")

        df = df[df['tokens_lemmatized'].apply(lambda x: bool(x) and len(x) > 0, meta=('tokens_lemmatized', bool))]
        linhas_excluidas = total_linhas - len(df)
        
        logger.info(f"Total de linhas após a remoção: {len(df)}")
        logger.info(f"Número de linhas excluídas: {linhas_excluidas}")
        
        return df
    except Exception as e:
        logger.error("Erro ao remover documentos vazios: %s", e)
        raise

# Função para classificar sentimentos usando VADER
analyzer = SentimentIntensityAnalyzer()

def classificar_sentimento_vader(texto):
    analise = analyzer.polarity_scores(texto)
    if analise['compound'] >= 0.05:
        return 'positivo'
    elif analise['compound'] <= -0.05:
        return 'negativo'
    else:
        return 'neutro'

# Função para classificar sentimentos usando BERT
bert_classifier = pipeline('sentiment-analysis')

def classificar_sentimento_bert(texto):
    resultado = bert_classifier(texto)[0]
    if resultado['label'] == 'POSITIVE':
        return 'positivo'
    elif resultado['label'] == 'NEGATIVE':
        return 'negativo'
    else:
        return 'neutro'

def combinar_sentimentos(sentimento_vader, sentimento_bert):
    if sentimento_vader == sentimento_bert:
        return sentimento_vader
    else:
        return sentimento_bert

def stemming_lemmatizacao():
    try:
        logger.info("Carregando o dataset com stopwords removidas.")
        stopwords_removed_data_path = os.path.join(config['directories']['processed_data'], config['files']['processed_dataset'])
        df = dd.read_parquet(stopwords_removed_data_path)
        logger.info("Dataset carregado com sucesso.")
        
        logger.info("Aplicando stemming.")
        df['tokens_stemmed'] = df['tokens_sem_stopwords'].apply(aplicar_stemming, meta=('tokens_stemmed', object))
        
        logger.info("Aplicando lematização.")
        df['tokens_lemmatized'] = df['tokens_sem_stopwords'].apply(aplicar_lemmatizacao, meta=('tokens_lemmatized', object))

        df = remover_documentos_vazios(df)
        
        # Adicionar colunas de sentimento usando VADER e BERT
        logger.info("Classificando sentimentos usando VADER e BERT.")
        df['sentimento_vader'] = df['tweet'].apply(classificar_sentimento_vader, meta=('sentimento_vader', object))
        df['sentimento_bert'] = df['tweet'].apply(classificar_sentimento_bert, meta=('sentimento_bert', object))

        # Combinar os sentimentos conforme a regra definida
        df['sentimento'] = df.apply(lambda row: combinar_sentimentos(row['sentimento_vader'], row['sentimento_bert']), axis=1, meta=('sentimento', object))

        # Codificar variáveis categóricas de sentimento
        logger.info("Codificando sentimentos.")
        df['sentimento_codificado'] = df.map_partitions(lambda part: pd.Series(LabelEncoder().fit(part['sentimento']).transform(part['sentimento'])), meta=('sentimento_codificado', int))

        df = remover_documentos_vazios(df)
        
        logger.info("Salvando dataset com stemming, lematização e classificação de sentimentos em formato Parquet e CSV.")
        stemmed_lemmatized_data_path = os.path.join(config['directories']['processed_data'], "etapa2_5_stemming_lemmatizacao.parquet")
        stemmed_lemmatized_data_csv_path = os.path.join(config['directories']['processed_data'], "etapa2_5_stemming_lemmatizacao.csv")
        
        df = df.compute()  # Convert Dask DataFrame to Pandas DataFrame
        df.to_csv(stemmed_lemmatized_data_csv_path, index=False)
        table = pa.Table.from_pandas(df)
        pq.write_table(table, stemmed_lemmatized_data_path)
        
        logger.info("Dataset salvo com sucesso.")
        
        # Gerar uma amostra do dataset para revisão manual
        logger.info("Gerando amostra do dataset.")
        amostra = df[['tweet', 'sentimento']].sample(n=30, random_state=42)
        amostra_path = os.path.join(config['directories']['processed_data'], "amostra_classificacao_sentimentos.csv")
        amostra.to_csv(amostra_path, index=False)
        logger.info(f"Amostra do dataset salva em: {amostra_path}")
        
        # Atualizar config.yaml para a próxima etapa
        atualizar_config(config, 'files', {'log_file': "etapa2_5_stemming_lemmatizacao.log",
                                           'processed_dataset': "etapa2_5_stemming_lemmatizacao.parquet",
                                           'processed_dataset_csv': "etapa2_5_stemming_lemmatizacao.csv",
                                           'raw_dataset': "asentimentos.parquet"})
    except Exception as e:
        logger.error("Erro ao aplicar stemming, lematização e classificação de sentimentos: %s", e)
        raise

if __name__ == "__main__":
    stemming_lemmatizacao()
