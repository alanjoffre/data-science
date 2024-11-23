import pandas as pd
import logging
import yaml
import os

# Carregar arquivo de configuração
config_path = 'D:\\Github\\data-science\\projetos\\analise-de-sentimentos\\nlp\\config\\config.yaml'
with open(config_path, 'r') as file:
    config = yaml.safe_load(file)

# Configuração de Logs
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[logging.StreamHandler(),
                              logging.FileHandler(os.path.join(config['directories']['logs'], "etapa2_4_remocao_stopwords.log"))])
logger = logging.getLogger(__name__)

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

def atualizar_config(config, chave, valor):
    config[chave] = valor
    with open(config_path, 'w') as file:
        yaml.safe_dump(config, file)
    logger.info("Arquivo config.yaml atualizado com sucesso.")

def remover_stopwords(tokens):
    return [token for token in tokens if token.lower() not in stopwords]

def remocao_stopwords():
    try:
        logger.info("Carregando o dataset tokenizado.")
        tokenized_data_path = os.path.join(config['directories']['processed_data'], config['files']['processed_dataset'])
        df = pd.read_parquet(tokenized_data_path)
        logger.info("Dataset carregado com sucesso.")
        
        logger.info("Removendo stopwords.")
        df['tokens'] = df['tokens'].apply(lambda x: x.split(', ') if isinstance(x, str) else x)  # Garantir que tokens sejam listas
        df['tokens_sem_stopwords'] = df['tokens'].apply(remover_stopwords)
        
        logger.info("Salvando dataset com stopwords removidas em formato Parquet e CSV.")
        stopwords_removed_data_path = os.path.join(config['directories']['processed_data'], "etapa2_4_remocao_stopwords.parquet")
        stopwords_removed_data_csv_path = os.path.join(config['directories']['processed_data'], "etapa2_4_remocao_stopwords.csv")
        
        # Converter os tokens para strings antes de salvar em CSV
        df['tokens_sem_stopwords'] = df['tokens_sem_stopwords'].apply(lambda x: ' '.join(x))
        df.to_csv(stopwords_removed_data_csv_path, index=False)
        df.to_parquet(stopwords_removed_data_path, index=False)
        
        logger.info("Dataset salvo com sucesso.")
        
        # Atualizar config.yaml para a próxima etapa
        atualizar_config(config, 'files', {'log_file': "etapa2_4_remocao_stopwords.log",
                                           'processed_dataset': "etapa2_4_remocao_stopwords.parquet",
                                           'processed_dataset_csv': "etapa2_4_remocao_stopwords.csv",
                                           'raw_dataset': "asentimentos.parquet"})
        
    except Exception as e:
        logger.error("Erro ao remover stopwords: %s", e)
        raise

if __name__ == "__main__":
    remocao_stopwords()
