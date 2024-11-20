import pandas as pd
import yaml
import logging
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.corpus import wordnet
import nltk

# Baixar recursos necessários do NLTK
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
        return wordnet.NOUN  # Default to noun if no outra POS is matched

# Carregar arquivo de configuração
config_path = 'D:\\Github\\data-science\\projetos\\analise-de-sentimentos\\nlp\\config\\config.yaml'
with open(config_path, 'r') as file:
    config = yaml.safe_load(file)

# Configuração de Logs
logging.basicConfig(level=logging.INFO)

def test_preprocessar_dados():
    processed_data_path = config['directories']['processed_data'] + 'etapa2_preprocessamento.parquet'
    try:
        df = pd.read_parquet(processed_data_path)
        assert 'processed_tweet' in df.columns, "Coluna 'processed_tweet' não encontrada"
        assert 'sentimento' in df.columns, "Coluna 'sentimento' não encontrada"
        assert df['processed_tweet'].str.islower().all(), "Nem todos os tweets foram convertidos para minúsculas"

        # Verificação da lematização
        lemmatizer = WordNetLemmatizer()
        def verificar_lemmatizacao(text):
            words = word_tokenize(text)
            pos_tags = nltk.pos_tag(words)
            lemmatized_words = [lemmatizer.lemmatize(word, get_wordnet_pos(tag)) for word, tag in pos_tags]
            return ' '.join(lemmatized_words) == text
        
        lemmatization_correct = df['processed_tweet'].apply(verificar_lemmatizacao).all()
        assert lemmatization_correct, "A lematização não foi aplicada corretamente"

        logging.info("Teste de pré-processamento de dados: SUCESSO!")
    except AssertionError as e:
        logging.error(f"Teste de pré-processamento de dados: FALHA! Erro: {e}")
    except Exception as e:
        logging.error("Teste de pré-processamento de dados: FALHA! Erro: %s", e, exc_info=True)

if __name__ == "__main__":
    test_preprocessar_dados()
