import pandas as pd
import os

# Configuração dos caminhos
dataset_path = 'D:/Github/data-science/projetos/logistica_transporte/2_analise_compartilhamento_de_viagem_classificacao/data/raw/logistica_transportadora_2018_2022.csv'
output_csv_path = 'D:/Github/data-science/projetos/logistica_transporte/2_analise_compartilhamento_de_viagem_classificacao/data/processed/preprocessado.csv'
output_parquet_path = 'D:/Github/data-science/projetos/logistica_transporte/2_analise_compartilhamento_de_viagem_classificacao/data/processed/preprocessado.parquet'

# Importando o dataset com tipo de dado especificado para a coluna 13
dtype_dict = {13: 'str'}  # Especificar a coluna 13 como string
dataset = pd.read_csv(dataset_path, dtype=dtype_dict, low_memory=False)

# Coletando 1% dos dados
sampled_data = dataset.sample(frac=0.01, random_state=42)

# Salvando os dados amostrados
os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
sampled_data.to_csv(output_csv_path, index=False)
sampled_data.to_parquet(output_parquet_path, index=False)

print("Amostra criada e salva com sucesso!")
