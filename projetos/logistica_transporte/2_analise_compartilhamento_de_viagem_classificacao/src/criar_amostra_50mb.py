import polars as pl
import os
import time

# Caminho dos arquivos
dataset_path = "D:/Github/data-science/projetos/logistica_transporte/2_analise_compartilhamento_de_viagem_classificacao/data/raw/logistica_transportadora_2018_2022.csv"
parquet_output = "D:/Github/data-science/projetos/logistica_transporte/2_analise_compartilhamento_de_viagem_classificacao/data/processed/preprocessado.parquet"
csv_output = "D:/Github/data-science/projetos/logistica_transporte/2_analise_compartilhamento_de_viagem_classificacao/data/processed/preprocessado.csv"

# Parâmetros de leitura
target_size_mb = 49   # Tamanho máximo da amostra
target_size_bytes = target_size_mb * 1024 * 1024  # Convertendo MB para Bytes

# Iniciar cronômetro
t_start = time.time()

# Leitura eficiente do dataset
df = pl.read_csv(dataset_path)

# Amostragem proporcional para manter a representatividade
total_size = df.estimated_size()
sample_fraction = min(1.0, target_size_bytes / total_size)  # Calcula a fração necessária para atingir 49MB
sampled_df = df.sample(fraction=sample_fraction, seed=42)

# Salvar a amostra em Parquet e CSV
sampled_df.write_parquet(parquet_output)
sampled_df.write_csv(csv_output)

# Verificar tamanho final
final_size_mb = os.path.getsize(parquet_output) / (1024 * 1024)

# Contar número de linhas da amostra
num_rows = sampled_df.height

# Tempo total de execução
t_end = time.time()
total_time = t_end - t_start
hours, rem = divmod(total_time, 3600)
minutes, seconds = divmod(rem, 60)

print(f"Amostra salva com sucesso! Tamanho final: {final_size_mb:.2f} MB")
print(f"Número de linhas na amostra: {num_rows}")
print(f"Tempo total do processo: {int(hours)}h {int(minutes)}m {seconds:.2f}s")