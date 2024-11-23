import pandas as pd
import yaml

# Caminho para o arquivo vetorizado
dataset_path = 'D:\\Github\\data-science\\projetos\\analise-de-sentimentos\\nlp\\data\\processed\\etapa2_6_vetorizacao_texto.parquet'

# Carregar o dataset vetorizado
df = pd.read_parquet(dataset_path)

# Calcular Estatísticas Descritivas
estatisticas_descritivas = df.describe()

# Salvar Estatísticas Descritivas em CSV
estatisticas_descritivas.to_csv('D:\\Github\\data-science\\projetos\\analise-de-sentimentos\\nlp\\data\\processed\\estatisticas_descritivas.csv')

# Salvar Estatísticas Descritivas em Parquet
estatisticas_descritivas.to_parquet('D:\\Github\\data-science\\projetos\\analise-de-sentimentos\\nlp\\data\\processed\\estatisticas_descritivas.parquet')

# Atualizar o arquivo de configuração
config = {
    'directories': {
        'processed_data': 'D:\\Github\\data-science\\projetos\\analise-de-sentimentos\\nlp\\data\\processed\\',
    },
    'files': {
        'dataset_path': 'etapa2_6_vetorizacao_texto.parquet',
        'estatisticas_descritivas_csv': 'estatisticas_descritivas.csv',
        'estatisticas_descritivas_parquet': 'estatisticas_descritivas.parquet'
    }
}

with open('D:\\Github\\data-science\\projetos\\analise-de-sentimentos\\nlp\\config\\config.yaml', 'w') as file:
    yaml.dump(config, file)

print("Análise Estatística concluída com sucesso.")
