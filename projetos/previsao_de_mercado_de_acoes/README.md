# Previsão de Mercado de Ações - Dólar/Real

Este projeto tem como objetivo a previsão do mercado de câmbio entre o dólar e o real utilizando diferentes técnicas de machine learning, como Regressão Linear, Random Forest, XGBoost, Redes Neurais e LSTM. A previsão é realizada utilizando dados históricos coletados através do Yahoo Finance, com diversas etapas de processamento, treinamento de modelos e avaliação de desempenho.

## Estrutura do Projeto

O projeto está organizado da seguinte forma:

D:/Github/data-science/projetos/previsao_de_mercado_de_acoes/random_forest/ │ ├── config/ # Arquivos de configuração (ex.: config.yaml) │ ├── data/ # Dados de entrada e resultados (ex.: dados de câmbio, predições) │ ├── raw/ # Dados brutos coletados │ ├── processed/ # Dados pré-processados │ ├── logs/ # Logs de execução do projeto │ ├── models/ # Modelos treinados e salvos │ ├── reports/ # Relatórios e gráficos │ └── figures/ # Gráficos gerados │ ├── src/ # Código-fonte │ ├── obter_dados_dia.py # Script para coleta de dados diários │ ├── pipeline.py # Pipeline de treinamento de modelos │ ├── 02_previsao_automatizada.py # Script para previsão automatizada │ └── requirements.txt # Arquivo de dependências


## Dependências

Este projeto utiliza as seguintes bibliotecas e ferramentas:

- Python 3.x
- `pandas` – Manipulação de dados
- `numpy` – Operações numéricas
- `yfinance` – Coleta de dados financeiros
- `sklearn` – Algoritmos de machine learning
- `xgboost` – Algoritmo de boosting
- `tensorflow` – Redes neurais e LSTM
- `statsmodels` – Modelos estatísticos (ARIMA, Exponential Smoothing)
- `arch` – Modelos GARCH
- `joblib` – Salvamento de modelos
- `matplotlib` – Visualização de gráficos
- `scipy` – Funções científicas e matemáticas
- `seaborn` – Visualização estatística

Você pode instalar todas as dependências utilizando o arquivo `requirements.txt`:
pip install -r requirements.txt

# Como Usar

## Passo 1: Coletando os Dados
O script obter_dados_dia.py coleta dados históricos de câmbio dólar/real (USDBRL=X) do Yahoo Finance, entre o período de 2010 até 2025. O script realiza o pré-processamento dos dados, incluindo o cálculo do logaritmo da coluna Close.

Execute o script da seguinte forma:
python src/obter_dados_dia.py

Os dados processados serão salvos no diretório data/raw/.

## Passo 2: Pipeline de Treinamento de Modelos
O script pipeline.py treina e avalia diferentes modelos de previsão, incluindo:

Regressão Linear
Random Forest
XGBoost
Redes Neurais Simples
LSTM
Modelos Estatísticos (ARIMA, GARCH, Exponential Smoothing)
Ele realiza os seguintes passos:

Coleta e pré-processamento de dados
Treinamento e avaliação de modelos com validação cruzada
Cálculo de métricas de avaliação como RMSE, MAE e R²
Salvamento do melhor modelo
Execute o script com o seguinte comando:
python src/pipeline.py

## Passo 3: Previsão Automatizada
O script 02_previsao_automatizada.py utiliza o melhor modelo treinado para fazer previsões automatizadas. Ele carrega o modelo salvo e realiza a previsão dos próximos valores de câmbio.

Execute o script da seguinte forma:
python src/02_previsao_automatizada.py

## Passo 4: Resultados
Os resultados do modelo treinado são salvos no diretório reports/figures/ como gráficos e no arquivo reports/figures/model_results.csv como um arquivo CSV com as métricas de desempenho.

# Arquivo de Configuração
O arquivo config/config.yaml contém configurações do projeto, como diretórios de entrada e saída.

Exemplo:
directories:
  data_raw: "D:/Github/data-science/projetos/previsao_de_mercado_de_acoes/random_forest/data/raw/"
  models: "D:/Github/data-science/projetos/previsao_de_mercado_de_acoes/random_forest/models/"
  preprocessors: "D:/Github/data-science/projetos/previsao_de_mercado_de_acoes/random_forest/preprocessors/"
  figures: "D:/Github/data-science/projetos/previsao_de_mercado_de_acoes/random_forest/reports/figures/"
  logs: "D:/Github/data-science/projetos/previsao_de_mercado_de_acoes/random_forest/logs/"
  config: "D:/Github/data-science/projetos/previsao_de_mercado_de_acoes/random_forest/config/"

# Logs
Os logs de execução são registrados no diretório logs/ e contêm informações detalhadas sobre o treinamento e avaliação dos modelos.

# Resultados dos Modelos
Os resultados da avaliação dos modelos incluem as métricas RMSE, MAE e R². O melhor modelo é escolhido com base no RMSE.

# Conclusão
Este projeto oferece uma solução completa para a previsão de câmbio dólar/real utilizando técnicas de machine learning e modelos estatísticos. Ele pode ser expandido para incluir mais dados ou outros modelos de previsão.

# Licença
Este projeto está licenciado sob a MIT License.