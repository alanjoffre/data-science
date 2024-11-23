import dask.dataframe as dd
import logging
import yaml
import os
import re
from symspellpy import SymSpell, Verbosity
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
                              logging.FileHandler(os.path.join(config['directories']['logs'], "etapa2_2_normalizacao_dados.log"))])
logger = logging.getLogger(__name__)

def atualizar_config(config, chave, valor):
    config[chave] = valor
    with open(config_path, 'w') as file:
        yaml.safe_dump(config, file)
    logger.info("Arquivo config.yaml atualizado com sucesso.")

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

def normalizar_dados():
    try:
        logger.info("Carregando o dataset pré-processado.")
        preprocessed_data_path = os.path.join(config['directories']['processed_data'], config['files']['processed_dataset'])
        df = dd.read_parquet(preprocessed_data_path)
        logger.info("Dataset carregado com sucesso.")
        
        logger.info("Normalizando os textos.")
        df['tweet'] = df['tweet'].map(normalizar_texto)
        
        logger.info("Salvando dataset normalizado em formato Parquet e CSV.")
        normalized_data_path = os.path.join(config['directories']['processed_data'], "etapa2_2_normalizacao_dados.parquet")
        normalized_data_csv_path = os.path.join(config['directories']['processed_data'], "etapa2_2_normalizacao_dados.csv")
        
        df.to_csv(normalized_data_csv_path, single_file=True)
        table = pa.Table.from_pandas(df.compute())
        pq.write_table(table, normalized_data_path)
        
        logger.info("Dataset salvo com sucesso.")
        
        # Atualizar config.yaml para a próxima etapa
        atualizar_config(config, 'files', {'log_file': "etapa2_2_normalizacao_dados.log",
                                           'processed_dataset': "etapa2_2_normalizacao_dados.parquet",
                                           'processed_dataset_csv': "etapa2_2_normalizacao_dados.csv",
                                           'raw_dataset': "asentimentos.parquet"})
                
    except Exception as e:
        logger.error("Erro ao normalizar os dados: %s", e)
        raise

if __name__ == "__main__":
    normalizar_dados()
