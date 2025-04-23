import pandas as pd

# Caminho do dataset processado
file_path = r'D:\Github\data-science\projetos\creditcard_fraud_detection\creditcard\data\processed\creditcard_processed.csv'

# Importar o dataset
df = pd.read_csv(file_path)

# Informar quantas linhas tem o dataset
total_rows = len(df)
print(f"Total de linhas no dataset: {total_rows}")

# Informar quantas linhas tem a classe 0 e classe 1
class_0_count = df[df['Class'] == 0].shape[0]
class_1_count = df[df['Class'] == 1].shape[0]
print(f"Total de linhas na classe 0: {class_0_count}")
print(f"Total de linhas na classe 1: {class_1_count}")

# Selecionar 50 linhas da classe 0 e 50 linhas da classe 1
class_0_sample = df[df['Class'] == 0].sample(n=50, random_state=42)
class_1_sample = df[df['Class'] == 1].sample(n=50, random_state=42)

# Concatenar as amostras selecionadas
sample_df = pd.concat([class_0_sample, class_1_sample])

# Salvar o dataset de amostra no mesmo diretório
sample_file_path = r'D:\Github\data-science\projetos\creditcard_fraud_detection\creditcard\data\processed\amostra_modelagem.csv'
sample_df.to_csv(sample_file_path, index=False)
print(f"Dataset de amostra salvo em: {sample_file_path}")
