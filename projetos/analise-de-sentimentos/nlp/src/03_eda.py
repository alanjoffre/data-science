import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from dateutil import parser
from wordcloud import WordCloud

# Configurações gerais
sns.set(style='whitegrid')
output_dir = r'D:\Github\data-science\projetos\analise-de-sentimentos\nlp\reports\figures'
data_path = r'D:\Github\data-science\projetos\analise-de-sentimentos\nlp\data\processed\final_data.parquet'

# Função para mapear fusos horários
def tzinfos(tzname, tzoffset):
    timezones = {
        'PDT': -7 * 3600,  # Horário de Verão do Pacífico
        'PST': -8 * 3600,  # Horário Padrão do Pacífico
        # Adicione mais fusos horários conforme necessário
    }
    return timezones.get(tzname)

# Carregar o dataset
df = pd.read_parquet(data_path)

# Corrigir a coluna de datas
df['date'] = df['date'].apply(lambda x: parser.parse(x, tzinfos=tzinfos))

# Criar o diretório de saída se não existir
os.makedirs(output_dir, exist_ok=True)

def save_plot(fig, filename):
    fig.savefig(os.path.join(output_dir, filename))
    plt.close(fig)

# 1. Distribuição de Sentimentos
fig, ax = plt.subplots()
sns.countplot(x='sentimento', data=df, ax=ax)
ax.set_title('Distribuição de Sentimentos')
save_plot(fig, 'distribuicao_sentimentos.png')

# 2. Sentimentos ao Longo do Tempo
fig, ax = plt.subplots()
df.set_index('date', inplace=True)
df.resample('M')['sentimento'].value_counts().unstack().plot(ax=ax)
ax.set_xlim(df.index.min(), df.index.max())
ax.set_title('Sentimentos ao Longo do Tempo')
save_plot(fig, 'sentimentos_tempo.png')

# 3. Tokens mais Comuns
fig, ax = plt.subplots()
tokens = df['tokens'].explode()
sns.countplot(y=tokens, order=tokens.value_counts().iloc[:20].index, ax=ax)
ax.set_title('Tokens Mais Comuns')
save_plot(fig, 'tokens_comuns.png')

# 4. Bigramas mais Comuns
fig, ax = plt.subplots()
bigrams = df['bigrams'].explode()
sns.countplot(y=bigrams, order=bigrams.value_counts().iloc[:20].index, ax=ax)
ax.set_title('Bigramas Mais Comuns')
save_plot(fig, 'bigramas_comuns.png')

# 5. Sentimento por Usuário
fig, ax = plt.subplots()
user_sentiment = df.groupby('username')['sentimento'].value_counts().unstack().fillna(0)
user_sentiment.plot(kind='bar', stacked=True, ax=ax)
ax.set_title('Sentimento por Usuário')
save_plot(fig, 'sentimento_usuario.png')

# 6. Sentimento por Consulta
fig, ax = plt.subplots()
query_sentiment = df.groupby('query')['sentimento'].value_counts().unstack().fillna(0)
query_sentiment.plot(kind='bar', stacked=True, ax=ax)
ax.set_title('Sentimento por Consulta')
save_plot(fig, 'sentimento_consulta.png')

# 7. Tokens sem Stopwords mais Comuns
fig, ax = plt.subplots()
tokens_sem_stopwords = df['tokens_sem_stopwords'].explode()
sns.countplot(y=tokens_sem_stopwords, order=tokens_sem_stopwords.value_counts().iloc[:20].index, ax=ax)
ax.set_title('Tokens Sem Stopwords Mais Comuns')
save_plot(fig, 'tokens_sem_stopwords.png')

# 8. Tokens Stemmed mais Comuns
fig, ax = plt.subplots()
tokens_stemmed = df['tokens_stemmed'].explode()
sns.countplot(y=tokens_stemmed, order=tokens_stemmed.value_counts().iloc[:20].index, ax=ax)
ax.set_title('Tokens Stemmed Mais Comuns')
save_plot(fig, 'tokens_stemmed.png')

# 9. Tokens Lemmatizados mais Comuns
fig, ax = plt.subplots()
tokens_lemmatizados = df['tokens_lemmatizados'].explode()
sns.countplot(y=tokens_lemmatizados, order=tokens_lemmatizados.value_counts().iloc[:20].index, ax=ax)
ax.set_title('Tokens Lemmatizados Mais Comuns')
save_plot(fig, 'tokens_lemmatizados.png')

# 10. Word Clouds para Sentimentos
sentimentos = df['sentimento'].unique()
for sentimento in sentimentos:
    fig, ax = plt.subplots()
    words = ' '.join(df[df['sentimento'] == sentimento]['tweet'])
    wordcloud = WordCloud(width=800, height=400, max_font_size=100, max_words=100, background_color='white').generate(words)
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis('off')
    ax.set_title(f'Nuvem de Palavras para Sentimento: {sentimento}')
    save_plot(fig, f'wordcloud_{sentimento}.png')

# 11. Tamanho dos Tweets
fig, ax = plt.subplots()
df['tweet_length'] = df['tweet'].apply(len)
sns.histplot(df['tweet_length'], bins=20, kde=True, ax=ax)
ax.set_title('Distribuição do Tamanho dos Tweets')
save_plot(fig, 'distribuicao_tamanho_tweets.png')

# 12. Correlação Entre Sentimento e Comprimento dos Tweets
fig, ax = plt.subplots()
sns.boxplot(x='sentimento', y='tweet_length', data=df, ax=ax)
ax.set_title('Correlação Entre Sentimento e Comprimento dos Tweets')
save_plot(fig, 'correlacao_sentimento_tamanho_tweets.png')

# 13. Frequência de Tokens por Sentimento
fig, axes = plt.subplots(nrows=1, ncols=len(sentimentos), figsize=(20, 5))
for ax, sentimento in zip(axes, sentimentos):
    tokens_sentimento = df[df['sentimento'] == sentimento]['tokens'].explode()
    sns.countplot(y=tokens_sentimento, order=tokens_sentimento.value_counts().iloc[:10].index, ax=ax)
    ax.set_title(f'Tokens Mais Comuns ({sentimento})')
save_plot(fig, 'tokens_comuns_por_sentimento.png')

# 14. Frequência de Bigramas por Sentimento
fig, axes = plt.subplots(nrows=1, ncols=len(sentimentos), figsize=(20, 5))
for ax, sentimento in zip(axes, sentimentos):
    bigrams_sentimento = df[df['sentimento'] == sentimento]['bigrams'].explode()
    sns.countplot(y=bigrams_sentimento, order=bigrams_sentimento.value_counts().iloc[:10].index, ax=ax)
    ax.set_title(f'Bigramas Mais Comuns ({sentimento})')
save_plot(fig, 'bigramas_comuns_por_sentimento.png')

print("Todos os gráficos foram gerados e salvos no diretório especificado.")
