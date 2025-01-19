from flask import Flask, request, render_template
import pandas as pd
import joblib
import yaml
import sys
import os

# Adicionar o caminho do diretório do arquivo `pipeline_de_producao.py` ao `sys.path`
sys.path.append('D:/Github/data-science/projetos/analise-de-sentimentos/nlp/src')

# Carregar configurações
config_path = 'D:/Github/data-science/projetos/analise-de-sentimentos/nlp/config/config.yaml'
with open(config_path, 'r') as file:
    config = yaml.safe_load(file)

# Inicializar Flask
app = Flask(__name__)

# Carregar o modelo Naive Bayes e o TfidfVectorizer
model_path = 'D:/Github/data-science/projetos/analise-de-sentimentos/nlp/models/best_model_nb.pkl'
vectorizer_path = 'D:/Github/data-science/projetos/analise-de-sentimentos/nlp/preprocessors/tfidf_vectorizer.pkl'
model = joblib.load(model_path)
vectorizer = joblib.load(vectorizer_path)

# Função para carregar as funções de processamento
from pipeline_de_producao import limpar_texto, classificar_sentimento_vader, classificar_sentimento_bert, classificar_sentimento_textblob

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        tweet = request.form['tweet']
        if tweet:
            tweet_limpo = limpar_texto(tweet)
            sentimento_vader = classificar_sentimento_vader(tweet_limpo)
            sentimento_bert = classificar_sentimento_bert(tweet_limpo)
            sentimento_textblob = classificar_sentimento_textblob(tweet_limpo)
            
            # Função para combinar os sentimentos
            def combinar_sentimentos(sent_vader, sent_bert, sent_textblob):
                if (sent_vader == sent_bert) or (sent_vader == sent_textblob):
                    return sent_vader
                elif sent_bert == sent_textblob:
                    return sent_bert
                else:
                    return sent_vader
            
            sentimento_final = combinar_sentimentos(sentimento_vader, sentimento_bert, sentimento_textblob)

            # Vectorize the cleaned tweet and predict with Naive Bayes model
            tweet_vec = vectorizer.transform([tweet_limpo]).toarray()
            sentimento_nb = model.predict(tweet_vec)[0]

            # Map the encoded sentiment to the corresponding text
            sentimento_mapeado = {0: 'negativo', 1: 'neutro', 2: 'positivo'}
            sentimento_nb_mapeado = sentimento_mapeado[sentimento_nb]

            return render_template('index.html', tweet=tweet, sentimento=sentimento_final, sentimento_nb=sentimento_nb_mapeado)

    return render_template('index.html')

if __name__ == "__main__":
    app.run(debug=True)
