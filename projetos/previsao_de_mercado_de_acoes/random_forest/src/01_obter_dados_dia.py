import os
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

# Diretório para salvar os dados
data_dir = "D:/Github/data-science/projetos/previsao_de_mercado_de_acoes/random_forest/data/raw/"
os.makedirs(data_dir, exist_ok=True)

# Definindo o período
start_date = '2010-01-01'
end_date = '2025-01-31'

# Coletando dados de câmbio dólar/real
ticker = 'USDBRL=X'
df = yf.download(ticker, start=start_date, end=end_date)
df.reset_index(inplace=True)
df.columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Adj_Close', 'Volume']

# Limpeza de dados
df.dropna(subset=['Open', 'High', 'Low', 'Close', 'Adj_Close'], inplace=True)
df.drop(columns='Volume', inplace=True)
df = df[(df['Open'] > 0) & (df['High'] > 0) & (df['Low'] > 0) & (df['Close'] > 0) & (df['Adj_Close'] > 0)]

# Garantir que o log seja calculado apenas para valores positivos
if (df['Close'] <= 0).any():
    print("Aviso: Alguns valores em 'Close' são <= 0 e serão ignorados no cálculo do log.")
    df = df[df['Close'] > 0]

# Cálculo do logaritmo
df['Log_Close'] = np.log(df['Close'])

# Exemplo de saída para validação
print(df[['Date', 'Close', 'Log_Close']].tail())

# Salvar os dados processados
data_path = os.path.join(data_dir, 'dados_processados.csv')
df.to_csv(data_path, index=False)
print(f"Dados processados salvos em {data_path}")
