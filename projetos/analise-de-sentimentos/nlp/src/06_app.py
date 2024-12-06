from flask import Flask, request, jsonify, render_template
import joblib
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from textblob import TextBlob
import os
import re
import emoji
from langdetect import detect, LangDetectException
from symspellpy import SymSpell, Verbosity
import spacy
from nltk.tokenize import word_tokenize
from nltk.util import ngrams
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from transformers import pipeline
import logging
import yaml

# Configurações e logs
config_path = 'D:\\Github\\data-science\\projetos\\analise-de-sentimentos\\nlp\\config\\config.yaml'
with open(config_path, 'r') as file:
    config = yaml.safe_load(file)

log_path = 'D:\\Github\\data-science\\projetos\\analise-de-sentimentos\\nlp\\logs\\pipeline_de_producao.log'
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[logging.StreamHandler(),
                              logging.FileHandler(log_path)])
logger = logging.getLogger(__name__)

# Inicializar SymSpell e spaCy
sym_spell = SymSpell(max_dictionary_edit_distance=2, prefix_length=7)
dictionary_path = "frequency_dictionary_en_82_765.txt"
bigram_path = "frequency_bigramdictionary_en_243_342.txt"
sym_spell.load_dictionary(dictionary_path, 0, 1)
sym_spell.load_bigram_dictionary(bigram_path, 0, 2)

nlp = spacy.load("en_core_web_sm")

# Carregar contrações de um arquivo
contractions = {}
with open('contractions.txt', 'r') as file:
    for line in file:
        contraido, expandido = line.strip().split(',')
        contractions[contraido] = expandido

# Funções de pré-processamento
def remover_caracteres_repetidos(texto):
    return re.sub(r'(.)\1{2,}', r'\1', texto)

def remover_palavras_estrangeiras(texto):
    try:
        if detect(texto) != 'en':
            return ''
    except LangDetectException:
        return ''
    return texto

def limpar_texto(texto):
    texto = re.sub(r"http\S+", "", texto)  # Remover links
    texto = re.sub(r"www\S+", "", texto)  # Remover links
    texto = re.sub(r"@\w+", "", texto)  # Remover menções
    texto = re.sub(r"#\w+", "", texto)  # Remover hashtags
    texto = emoji.replace_emoji(texto, replace='')  # Remover emojis
    texto = re.sub(r"\d+", "", texto)  # Remover números
    texto = remover_caracteres_repetidos(texto)  # Remover caracteres repetidos
    texto = texto.strip()  # Remover espaços extras
    texto = texto.lower()  # Converter para minúsculas
    texto = remover_palavras_estrangeiras(texto)  # Remover palavras não inglesas
    texto = expandir_contracoes(texto)  # Expandir contrações
    return texto

def expandir_contracoes(texto):
    for contraido, expandido in contractions.items():
        texto = re.sub(fr"\b{contraido}\b", expandido, texto)
    return texto

def corrigir_ortografia(texto):
    suggestions = sym_spell.lookup_compound(texto, max_edit_distance=2)
    if suggestions:
        return suggestions[0].term
    return texto

def normalizar_texto(texto):
    texto = corrigir_ortografia(texto)
    texto = expandir_contracoes(texto)
    return texto

def tokenizar_texto(texto):
    doc = nlp(texto)
    tokens = [token.text for token in doc if not token.is_punct]
    return tokens

def criar_bigramas(texto):
    tokens = word_tokenize(texto)
    bigrams = list(ngrams(tokens, 2))
    return ["_".join(bigrama) for bigrama in bigrams]

stopwords = set([
    'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your', 'yours', 'yourself', 'yourselves',
    'he', 'him', 'his', 'himself', 'she', 'her', 'hers', 'herself', 'it', 'its', 'itself', 'they', 'them', 'their',
    'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', 'these', 'those', 'am', 'is', 'are',
    'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an',
    'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about',
    'against', 'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up',
    'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when',
    'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor',
    'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', 'should', 'now'
])

def remover_stopwords(tokens):
    return [token for token in tokens if token.lower() not in stopwords]

def aplicar_stemming(tokens):
    stemmer = SnowballStemmer(language='english')
    return [stemmer.stem(token) for token in tokens]

def aplicar_lemmatizacao(tokens):
    doc = nlp(" ".join(tokens))
    return [token.lemma_.strip() for token in doc if token.lemma_.strip()]

def classificar_sentimento_vader(texto):
    analyzer = SentimentIntensityAnalyzer()
    analise = analyzer.polarity_scores(texto)
    if analise['compound'] >= 0.05:
        return 'positivo'
    elif analise['compound'] <= -0.05:
        return 'negativo'
    else:
        return 'neutro'

bert_classifier = pipeline('sentiment-analysis', model="nlptown/bert-base-multilingual-uncased-sentiment")

def classificar_sentimento_bert(texto):
    resultado = bert_classifier(texto)[0]
    if resultado['label'] == '5 stars':
        return 'positivo'
    elif resultado['label'] == '1 star':
        return 'negativo'
    else:
        return 'neutro'

def combinar_sentimentos(sentimento_vader, sentimento_bert):
    if sentimento_vader == sentimento_bert:
        return sentimento_vader
    else:
        return sentimento_bert

# Carregar o modelo e o vetorizador
model_path = r'D:\Github\data-science\projetos\analise-de-sentimentos\nlp\models\best_ridge_model.pkl'
vectorizer_path = r'D:\Github\data-science\projetos\analise-de-sentimentos\nlp\preprocessors\tfidf_vectorizer.pkl'
model = joblib.load(model_path)
vectorizer = joblib.load(vectorizer_path)

# Inicializar a aplicação Flask
app = Flask(__name__)

# Rota para a página principal
@app.route('/')
def home():
    return render_template('index.html')

# Endpoint para realizar a predição
@app.route('/predict', methods=['POST'])
def predict():
    # Receber os dados do cliente
    data = request.form['tweet']
    tweet = data

    # Etapa 1: Limpar os dados
    tweet_limpo = limpar_texto(tweet)

    # Etapa 2: Normalizar o texto
    tweet_normalizado = normalizar_texto(tweet_limpo)

    # Etapa 3: Tokenizar e remover stopwords
    tokens = tokenizar_texto(tweet_normalizado)
    tokens_sem_stopwords = remover_stopwords(tokens)
    bigrams = criar_bigramas(tweet_normalizado)
    tokens_stemmed = aplicar_stemming(tokens_sem_stopwords)
    tokens_lemmatizados = aplicar_lemmatizacao(tokens_sem_stopwords)

    # Extrair features para a predição
    tweet_tfidf = vectorizer.transform([tweet_normalizado]).toarray()
    word_count = len(tokens)
    char_count = len(tweet_normalizado)
    avg_word_length = np.mean([len(word) for word in tokens])
    sentiment = TextBlob(tweet_normalizado).sentiment.polarity
    features = np.hstack((tweet_tfidf, np.array([[word_count, char_count, avg_word_length, sentiment]])))

    # Realizar a predição
    prediction = model.predict(features)

    # Classificação de sentimentos
    sentimento_vader = classificar_sentimento_vader(tweet_normalizado)
    sentimento_bert = classificar_sentimento_bert(tweet_normalizado)
    sentimento_final = combinar_sentimentos(sentimento_vader, sentimento_bert)

    # Mapear a predição para um sentimento
    sentiment_label = {0: 'negativo', 1: 'neutro', 2: 'positivo'}
    result = sentiment_label[prediction[0]]

    # Salvar a predição em um arquivo CSV
    prediction_data = pd.DataFrame({
        'tweet': [tweet],
        'sentimento_real': [result]
    })
    predictions_dir = 'D:\\Github\\data-science\\projetos\\analise-de-sentimentos\\nlp\\predictions'
    if not os.path.exists(predictions_dir):
        os.makedirs(predictions_dir)
    predictions_file = os.path.join(predictions_dir, 'predictions.csv')

    if os.path.exists(predictions_file):
        prediction_data.to_csv(predictions_file, mode='a', header=False, index=False)
    else:
        prediction_data.to_csv(predictions_file, index=False)

    return render_template('index.html', tweet=tweet, prediction=result)

if __name__ == '__main__':
    app.run(debug=True)
