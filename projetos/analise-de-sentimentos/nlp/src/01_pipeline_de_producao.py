import dask.dataframe as dd
import pandas as pd
import logging
import yaml
import os
import pyarrow as pa
import pyarrow.parquet as pq
import re
import emoji
from langdetect import detect, LangDetectException
import time

# Carregar arquivo de configuração
config_path = 'D:\\Github\\data-science\\projetos\\analise-de-sentimentos\\nlp\\config\\config.yaml'
with open(config_path, 'r') as file:
    config = yaml.safe_load(file)

# Configuração de Logs
log_path = 'D:\\Github\\data-science\\projetos\\analise-de-sentimentos\\nlp\\logs\\pipeline_de_producao.log'
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[logging.StreamHandler(),
                              logging.FileHandler(log_path)])
logger = logging.getLogger(__name__)

def carregar_dados():
    try:
        logger.info("Carregando o dataset bruto.")
        raw_data_path = 'D:\\Github\\data-science\\projetos\\analise-de-sentimentos\\nlp\\data\\raw\\asentimentos.parquet'
        df = dd.read_parquet(raw_data_path)
        total_linhas = len(df)
        logger.info(f"Dataset carregado com sucesso com {total_linhas} linhas.")
        
        logger.info("Renomeando colunas.")
        colunas_novas = ["index", "id", "date", "query", "username", "tweet"]
        df.columns = colunas_novas
        
        return df
    except Exception as e:
        logger.error("Erro ao carregar os dados: %s", e)
        raise
    
def remover_caracteres_repetidos(texto):
    """
    Remove caracteres repetidos no texto, mantendo apenas um exemplo de cada.
    """
    return re.sub(r'(.)\1{2,}', r'\1', texto)

def remover_palavras_estrangeiras(texto):
    """
    Remove palavras que não estão em inglês.
    """
    try:
        if detect(texto) != 'en':
            return ''
    except LangDetectException:
        return ''
    return texto

def limpar_texto(texto):
    """
    Função para limpar o texto removendo links, menções, hashtags, emojis, números, caracteres especiais,
    e palavras não inglesas, além de normalizar caracteres repetidos e manter emoticons simples.
    """
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

def preprocessar_dados(df):
    """
    Função para pré-processar os dados aplicando a função de limpeza de texto.
    """
    try:
        logger.info("Pré-processando os dados.")
        df['tweet'] = df['tweet'].apply(limpar_texto)
        return df
    except Exception as e:
        logger.error("Erro ao preprocessar os dados: %s", e)
        raise

from symspellpy import SymSpell, Verbosity

# Inicializar SymSpell
sym_spell = SymSpell(max_dictionary_edit_distance=2, prefix_length=7)
dictionary_path = "frequency_dictionary_en_82_765.txt"
bigram_path = "frequency_bigramdictionary_en_243_342.txt"
sym_spell.load_dictionary(dictionary_path, 0, 1)
sym_spell.load_bigram_dictionary(bigram_path, 0, 2)

# Carregar contrações de um arquivo
contractions = {}
with open('contractions.txt', 'r') as file:
    for line in file:
        contraido, expandido = line.strip().split(',')
        contractions[contraido] = expandido

def corrigir_ortografia(texto):
    suggestions = sym_spell.lookup_compound(texto, max_edit_distance=2)
    if suggestions:
        return suggestions[0].term
    return texto

def expandir_contracoes(texto):
    for contraido, expandido in contractions.items():
        texto = re.sub(fr"\b{contraido}\b", expandido, texto)
    return texto

def normalizar_texto(texto):
    texto = corrigir_ortografia(texto)
    texto = expandir_contracoes(texto)
    return texto

def normalizar_dados(df):
    """
    Função para normalizar os dados aplicando correção ortográfica e expansão de contrações.
    """
    try:
        logger.info("Normalizando os textos.")
        df['tweet'] = df['tweet'].apply(normalizar_texto)
        return df
    except Exception as e:
        logger.error("Erro ao normalizar os dados: %s", e)
        raise

import spacy
from nltk.tokenize import word_tokenize
from nltk.util import ngrams

# Carregar modelo de linguagem spaCy
nlp = spacy.load("en_core_web_sm")

def tokenizar_texto(texto):
    """
    Tokeniza um texto usando spaCy, removendo pontuações.
    """
    doc = nlp(texto)
    tokens = [token.text for token in doc if not token.is_punct]  # Filtra pontuação
    return tokens  # Retorna uma lista de tokens sem pontuação

def criar_bigramas(texto):
    """
    Cria bigramas a partir de um texto.
    """
    tokens = word_tokenize(texto)
    bigrams = list(ngrams(tokens, 2))
    return ["_".join(bigrama) for bigrama in bigrams]

def tokenizacao_dados(df):
    """
    Realiza a tokenização de dados e salva o resultado.
    """
    try:
        logger.info("Tokenizando os textos.")
        df['tokens'] = df['tweet'].apply(tokenizar_texto)
        logger.info("Criando bigramas.")
        df['bigrams'] = df['tweet'].apply(criar_bigramas)
        return df
    except Exception as e:
        logger.error("Erro ao tokenizar os dados: %s", e)
        raise

from nltk.stem import SnowballStemmer

# Lista de stopwords em inglês (exemplo básico)
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

def remocao_stopwords(df):
    """
    Realiza a remoção de stopwords dos tokens e salva o resultado.
    """
    try:
        logger.info("Removendo stopwords.")
        df['tokens'] = df['tokens'].apply(lambda x: x if isinstance(x, list) else x.split(', '))
        df['tokens_sem_stopwords'] = df['tokens'].apply(remover_stopwords)
        return df
    except Exception as e:
        logger.error("Erro ao remover stopwords: %s", e)
        raise

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
        df = df[df['tokens_lemmatizados'].apply(lambda x: bool(x) and len(x) > 0)]
        return df
    except Exception as e:
        logger.error("Erro ao remover documentos vazios: %s", e)
        raise

def stemming_lemmatizacao(df):
    """
    Aplica stemming e lematização aos tokens e salva o resultado.
    """
    try:
        logger.info("Aplicando stemming.")
        df['tokens_stemmed'] = df['tokens_sem_stopwords'].apply(aplicar_stemming)
        
        logger.info("Aplicando lematização.")
        df['tokens_lemmatizados'] = df['tokens_sem_stopwords'].apply(aplicar_lemmatizacao)

        df = remover_documentos_vazios(df)
        
        return df
    except Exception as e:
        logger.error("Erro ao aplicar stemming e lematização: %s", e)
        raise

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from transformers import pipeline

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

def classificar_sentimentos(df):
    """
    Classifica sentimentos usando VADER e BERT, e combina os resultados.
    """
    try:
        logger.info("Classificando sentimentos usando VADER e BERT.")
        df['sentimento_vader'] = df['tweet'].apply(classificar_sentimento_vader)
        df['sentimento_bert'] = df['tweet'].apply(classificar_sentimento_bert)

        # Aplicar a regra para a coluna 'sentimento'
        df['sentimento'] = df.apply(lambda row: combinar_sentimentos(row['sentimento_vader'], row['sentimento_bert']), axis=1)

        return df
    except Exception as e:
        logger.error("Erro ao classificar sentimentos: %s", e)
        raise

from sklearn.preprocessing import LabelEncoder

def codificar_sentimentos(df):
    """
    Codifica variáveis categóricas de sentimento.
    Se os valores já forem numéricos, a função retorna o DataFrame sem alterações.
    """
    try:
        logger.info("Codificando sentimentos.")
        
        # Verifica se a coluna 'sentimento' já contém valores numéricos
        if df['sentimento'].dtype == 'object':  # Se os sentimentos são strings
            label_encoder = LabelEncoder()
            df['sentimento_codificado'] = label_encoder.fit_transform(df['sentimento'])
        else:
            logger.warning("A coluna 'sentimento' já contém valores numéricos. Nenhuma codificação necessária.")
            df['sentimento_codificado'] = df['sentimento']  # Copia os valores sem alterações

        return df
    except Exception as e:
        logger.error("Erro ao codificar sentimentos: %s", e)
        raise

import json
from sklearn.feature_extraction.text import TfidfVectorizer

def contar_sentimentos(df):
    """
    Conta o número de sentimentos positivos, neutros e negativos.
    """
    try:
        logger.info("Contando sentimentos positivos, neutros e negativos.")
        contagem_sentimentos = df['sentimento'].value_counts()
        logger.info(f"Sentimentos positivos: {contagem_sentimentos.get('positivo', 0)}")
        logger.info(f"Sentimentos neutros: {contagem_sentimentos.get('neutro', 0)}")
        logger.info(f"Sentimentos negativos: {contagem_sentimentos.get('negativo', 0)}")
    except Exception as e:
        logger.error("Erro ao contar sentimentos: %s", e)
        raise

def salvar_dados(df):
    """
    Salva o dataset processado em formato Parquet e CSV seguindo o config.yaml.
    """
    try:
        output_path_parquet = os.path.join(config['directories']['processed_data'], 'final_data.parquet')
        output_path_csv = os.path.join(config['directories']['processed_data'], 'final_data.csv')
        amostra_path_parquet = os.path.join(config['directories']['processed_data'], 'amostra.parquet')
        amostra_path_csv = os.path.join(config['directories']['processed_data'], 'amostra.csv')
        
        logger.info("Salvando dataset processado.")
        
        # Converter para Pandas DataFrame se for um Dask DataFrame
        if isinstance(df, dd.DataFrame):
            df = df.compute()
        
        # Verificação de valores NaN em 'sentimento_codificado'
        nan_count = df['sentimento_codificado'].isna().sum()
        logger.info(f"Número de linhas com NaN em 'sentimento_codificado' antes da exclusão: {nan_count}")
        
        df = df.dropna(subset=['sentimento_codificado'])
        
        # Informar o número de linhas restantes após a remoção de NaNs
        linhas_restantes = len(df)
        logger.info(f"Número de linhas restantes após a remoção de NaNs: {linhas_restantes}")

        # Contar sentimentos antes de salvar
        contar_sentimentos(df)

        # Salvar uma amostra com 30 linhas contendo apenas as colunas 'tweet' e 'sentimento'
        amostra = df[['tweet', 'sentimento_vader', 'sentimento_bert', 'sentimento']].sample(n=30)
        amostra.to_csv(amostra_path_csv, index=False)
        amostra_table = pa.Table.from_pandas(amostra)
        pq.write_table(amostra_table, amostra_path_parquet)
        logger.info("Amostra salva com sucesso.")

        # Salvar o dataset completo
        df.to_csv(output_path_csv, index=False)
        table = pa.Table.from_pandas(df)
        pq.write_table(table, output_path_parquet)
        
        logger.info("Dataset salvo com sucesso.")
    except Exception as e:
        logger.error("Erro ao salvar os dados: %s", e)
        raise

def gerar_relatorios(df):
    """
    Gera relatórios estatísticos e gráficos.
    """
    try:
        logger.info("Gerando relatórios.")
        figures_path = os.path.join(config['directories']['reports'])
        
        if not os.path.exists(figures_path):
            os.makedirs(figures_path)

        # Converter para Pandas DataFrame se for um Dask DataFrame
        if isinstance(df, dd.DataFrame):
            df = df.compute()

        # Estatísticas Descritivas
        estatisticas_descritivas = df.describe()
        relatorio_estatisticas_path = os.path.join(figures_path, 'estatisticas_descritivas.csv')
        estatisticas_descritivas.to_csv(relatorio_estatisticas_path)
        
        logger.info("Relatórios gerados com sucesso.")
    except Exception as e:
        logger.error("Erro ao gerar relatórios: %s", e)
        raise

def processar_em_blocos(df, bloco_tamanho=5000):
    """
    Processa os dados em blocos e salva arquivos Parquet temporários no diretório temp dentro de processed_data.
    """
    try:
        total_linhas = len(df)
        logger.info(f"Total de linhas a serem processadas: {total_linhas}")
        total_blocos = (total_linhas + bloco_tamanho - 1) // bloco_tamanho
        logger.info(f"Total de blocos a serem processados: {total_blocos}")

        temp_data_dir = os.path.join(config['directories']['processed_data'], 'temp')
        
        # Criar diretório temporário se não existir
        if not os.path.exists(temp_data_dir):
            os.makedirs(temp_data_dir)
        
        blocos = []
        tempos_blocos = []
        for i in range(0, total_linhas, bloco_tamanho):
            bloco_inicio = time.time()  # Iniciar medição de tempo
            df_bloco = df.loc[i:i+bloco_tamanho-1].compute()
            logger.info(f"Processando bloco {i//bloco_tamanho + 1} de {total_blocos}")

            df_bloco = preprocessar_dados(df_bloco)
            df_bloco = normalizar_dados(df_bloco)
            df_bloco = tokenizacao_dados(df_bloco)
            df_bloco = remocao_stopwords(df_bloco)
            df_bloco = stemming_lemmatizacao(df_bloco)
            df_bloco = classificar_sentimentos(df_bloco)
            df_bloco = codificar_sentimentos(df_bloco)
            df_bloco = remover_documentos_vazios(df_bloco)
            
            bloco_fim = time.time()  # Finalizar medição de tempo
            tempos_blocos.append(bloco_fim - bloco_inicio)  # Registrar tempo do bloco
            
            bloco_path = os.path.join(temp_data_dir, f'bloco_{i//bloco_tamanho + 1}.parquet')
            table = pa.Table.from_pandas(df_bloco)
            pq.write_table(table, bloco_path)
            blocos.append(bloco_path)
            logger.info(f"Bloco {i//bloco_tamanho + 1} salvo em {bloco_path}")

            if i == 0:  # Estimar tempo total após o primeiro bloco
                tempo_medio_por_bloco = tempos_blocos[0]
                tempo_total_estimado = tempo_medio_por_bloco * total_blocos
                minutos, segundos = divmod(tempo_medio_por_bloco, 60)
                horas, minutos = divmod(minutos, 60)
                logger.info(f"Tempo médio por bloco: {int(horas)} horas, {int(minutos)} minutos, {segundos:.2f} segundos")
                minutos, segundos = divmod(tempo_total_estimado, 60)
                horas, minutos = divmod(minutos, 60)
                logger.info(f"Tempo total estimado: {int(horas)} horas, {int(minutos)} minutos, {segundos:.2f} segundos")

        # Calcular tempo médio por bloco e tempo total estimado novamente após todos os blocos processados
        tempo_medio_por_bloco = sum(tempos_blocos) / len(tempos_blocos)
        tempo_total_estimado = tempo_medio_por_bloco * total_blocos
        minutos, segundos = divmod(tempo_medio_por_bloco, 60)
        horas, minutos = divmod(minutos, 60)
        logger.info(f"Tempo médio por bloco (recalculado): {int(horas)} horas, {int(minutos)} minutos, {segundos:.2f} segundos")
        minutos, segundos = divmod(tempo_total_estimado, 60)
        horas, minutos = divmod(minutos, 60)
        logger.info(f"Tempo total estimado (recalculado): {int(horas)} horas, {int(minutos)} minutos, {segundos:.2f} segundos")

        df_combined = dd.read_parquet(blocos)
        salvar_dados(df_combined)
        logger.info("Todos os blocos processados e combinados com sucesso.")
    except Exception as e:
        logger.error("Erro ao processar os dados em blocos: %s", e)
        raise

if __name__ == "__main__":
    try:
        df = carregar_dados()
        processar_em_blocos(df)
        gerar_relatorios(df)
    except Exception as e:
        logger.error("Erro durante a execução do pipeline de produção: %s", e)
