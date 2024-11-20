import dask.dataframe as dd
import logging
import yaml
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from textblob import TextBlob
import nltk
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import wordnet

# Baixar recursos necessários do NLTK
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')  # Para as POS tags

# Função para mapear tags POS para o formato necessário pelo lematizador
def get_wordnet_pos(tag):
    if tag.startswith('V'):
        return wordnet.VERB
    elif tag.startswith('N'):
        return wordnet.NOUN
    elif tag.startswith('R'):
        return wordnet.ADJ
    else:
        return wordnet.NOUN  # Default to noun if no other POS is matched

# Carregar arquivo de configuração
config_path = 'D:\\Github\\data-science\\projetos\\analise-de-sentimentos\\nlp\\config\\config.yaml'
with open(config_path, 'r') as file:
    config = yaml.safe_load(file)

# Configuração de Logs
logging.basicConfig(level=config['logs']['level'],
                    format=config['logs']['format'],
                    handlers=[logging.StreamHandler(),
                              logging.FileHandler(config['directories']['logs'] + 'preprocessamento_dados.log')])
logger = logging.getLogger(__name__)

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def limpar_dados(df):
    """Remove duplicatas e valores ausentes"""
    df = df.drop_duplicates()
    df = df.dropna()
    return df

def normalizar_texto(text):
    """Normaliza texto, convertendo para minúsculas e removendo pontuação, números e caracteres especiais"""
    text = text.lower()  # Converter para minúsculas
    text = re.sub(r'[^\w\s]', '', text)  # Remover pontuação
    text = re.sub(r'\d+', '', text)  # Remover números
    return text

def remover_stop_words(text):
    """Remove stop words do texto"""
    words = text.split()
    words = [word for word in words if word not in stop_words]
    return ' '.join(words)

def aplicar_lemmatizacao(text):
    """Aplica lematização ao texto, usando POS tags para maior precisão"""
    words = word_tokenize(text)
    pos_tags = nltk.pos_tag(words)
    lemmatized_words = [lemmatizer.lemmatize(word, get_wordnet_pos(tag)) for word, tag in pos_tags]
    return ' '.join(lemmatized_words)

def analisar_sentimento(text):
    """Classifica o sentimento de um texto como positivo, negativo ou neutro"""
    blob = TextBlob(text)
    polaridade = blob.sentiment.polarity
    if polaridade > 0:
        return 'positivo'
    elif polaridade < 0:
        return 'negativo'
    else:
        return 'neutro'

def preprocessar_texto(text):
    """Aplica todas as etapas de pré-processamento"""
    normalized_text = normalizar_texto(text)
    text_without_stopwords = remover_stop_words(normalized_text)
    lemmatized_text = aplicar_lemmatizacao(text_without_stopwords)
    sentimento = analisar_sentimento(lemmatized_text)
    return lemmatized_text, sentimento

def preprocessar_dados():
    # Carregar o dataset processado da etapa anterior
    processed_data_path = config['directories']['processed_data'] + config['files']['processed_dataset']
    logger.info("Carregando dataset processado do caminho: %s", processed_data_path)

    # Carregar o DataFrame usando Dask
    try:
        df = dd.read_parquet(processed_data_path)
    except Exception as e:
        logger.error(f"Erro ao carregar o dataset: {e}")
        return

    logger.info("Dataset carregado com sucesso!")

    # Limpeza de dados
    df = limpar_dados(df)
    logger.info("Dados limpos com sucesso!")

    # Verificar se a coluna 'tweet' existe
    if 'tweet' not in df.columns:
        logger.error("A coluna 'tweet' não está presente no dataset.")
        return

    # Aplicar pré-processamento e classificar sentimentos
    def process_and_classify(tweet):
        processed_text, sentimento = preprocessar_texto(tweet)
        return processed_text, sentimento

    # Aplicar a função de processamento e classificação separadamente para cada saída
    processed_results = df['tweet'].map(process_and_classify, meta=('result', 'object'))
    df['processed_tweet'] = processed_results.map(lambda x: x[0], meta=('processed_tweet', 'object'))
    df['sentimento'] = processed_results.map(lambda x: x[1], meta=('sentimento', 'str'))

    logger.info("Texto pré-processado e sentimentos classificados com sucesso!")

    # Converter DataFrame Dask em pandas para salvar e imprimir
    df_pd = df.compute()

    # Salvar o dataset completo com colunas adicionais
    processed_data_parquet = config['directories']['processed_data'] + 'etapa2_preprocessamento.parquet'
    processed_data_csv = config['directories']['processed_data'] + 'etapa2_preprocessamento.csv'
    df_pd.to_parquet(processed_data_parquet, engine='pyarrow', index=False)
    df_pd.to_csv(processed_data_csv, index=False)
    logger.info("Dataset pré-processado completo salvo em %s e %s", processed_data_parquet, processed_data_csv)

    # Atualizar o arquivo de configuração
    config['files']['processed_dataset'] = 'etapa2_preprocessamento.parquet'
    with open(config_path, 'w') as file:
        yaml.dump(config, file)
    logger.info("Arquivo de configuração atualizado com sucesso.")

if __name__ == "__main__":
    preprocessar_dados()
