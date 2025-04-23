import pandas as pd
import yaml
import logging
import time
from sklearn.preprocessing import StandardScaler

# Registrar a hora inicial do processamento
start_time = time.time()

# Configurar logging
with open(r'D:\Github\data-science\projetos\creditcard_fraud_detection\creditcard\config\config.yaml', 'r') as file:
    config = yaml.safe_load(file)

logging.basicConfig(filename=config['paths']['logs_path'] + 'feature_engineering.log', level=logging.INFO, format='%(asctime)s - %(message)s')

# Carregar os dados processados
df = pd.read_csv(config['data']['processed_data_path'])

# Criação de novos atributos
logging.info("Criando novos atributos.")

# Número de transações por hora
df['Hour'] = (df['Time'] // 3600) % 24

# Exibir as primeiras linhas com o novo atributo
logging.info("Primeiras linhas do dataframe com novos atributos:")
logging.info(df.head())

# Seleção de recursos
logging.info("Selecionando recursos.")

# Incluindo o novo atributo 'Hour' na lista de features
features = df.drop(columns=['Class', 'Time'])

# Normalizar os dados novamente
scaler = StandardScaler()
features_normalized = scaler.fit_transform(features)
df_normalized = pd.DataFrame(features_normalized, columns=features.columns)
df_normalized['Class'] = df['Class']

# Salvar o dataset processado novamente com os novos atributos
df_normalized.to_csv(config['data']['processed_data_path'], index=False)
logging.info("Dataset processado salvo com novos atributos em creditcard_processed.csv")

# Registrar a hora final do processamento
end_time = time.time()
total_time = end_time - start_time
hours, remainder = divmod(total_time, 3600)
minutes, seconds = divmod(remainder, 60)
logging.info(f"Tempo total de processamento: {int(hours)} horas, {int(minutes)} minutos e {int(seconds)} segundos.")

print("3 - Etapa de Engenharia de Atributos e Seleção de Recursos concluída.")
