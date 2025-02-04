# Previsão de Gorjetas – Pipeline e Modelagem de Dados.
 - Este repositório contém um conjunto de scripts e pipelines para o processamento de dados, treinamento, avaliação e deploy de modelos de machine learning aplicados à previsão de gorjetas em operações de logística e transporte. Os processos englobam desde a preparação dos dados até a disponibilização de uma API em Flask para servir as previsões, com monitoramento via Prometheus.

 ## Visão Geral do Projeto
- O projeto foi desenvolvido utilizando Python e diversas bibliotecas de data science e machine learning (como scikit-learn, XGBoost, Optuna, entre outras), com o objetivo de criar soluções escaláveis e de alta performance para previsão de gorjetas. A estrutura do projeto contempla:

- Pré-processamento e Engenharia de Features: Conversão e tratamento de colunas (datas, numéricas e categóricas), criação de novas variáveis e amostragem de dados.

- Treinamento de Modelos: Implementação e avaliação de múltiplos algoritmos (Regressão Linear, Random Forest, Gradient Boosting, KNeighbors, SVR e XGBoost) com técnicas de ajuste de hiperparâmetros (Grid Search e Random Search).

- Otimização de Pipeline com Optuna: Pipeline otimizado que integra pré-processamento e modelagem, validado via validação cruzada.

- Deploy e API: Uma aplicação Flask que carrega o pipeline treinado, disponibilizando uma rota para previsões e outra para métricas expostas ao Prometheus.

# Estrutura do Repositório

- A seguir, uma breve descrição dos principais módulos:

- pipeline.py:
- Realiza a carga do dataset raw no formato Parquet.
- Executa transformações de dados, como conversão de datas, criação de features temporais (ano, mês, dia, hora e dia da semana) e tratamento de valores ausentes (preenchimento com média, mediana e moda).
- Amostragem dos dados e aplicação de um pipeline de pré-processamento (usando StandardScaler e OneHotEncoder).
- Treinamento de um modelo de RandomForestRegressor com barra de progresso (usando tqdm) e extração das importâncias das features.
- Geração e salvamento de visualizações (scatter plot e heatmap) para análise exploratória.

- treinando_modelos.py:
- Importa o dataset processado e seleciona amostras para treinamento.
- Realiza a divisão entre conjunto de treinamento e teste.
- Treina e avalia diversos modelos (Linear Regression, Random Forest, Gradient Boosting, KNeighbors, SVR, XGBoost) utilizando métricas como RMSE, MAE e R².
- Implementa ajustes de hiperparâmetros via RandomizedSearchCV e GridSearchCV para os modelos Random Forest e 
Gradient Boosting.

- modelo_previsao.py:
- Similar ao script anterior, mas foca em processos de tuning e avaliação detalhada dos modelos de regressão para previsão de gorjetas.
- Utiliza uma amostra de 10% dos dados e gera um comparativo de performance dos algoritmos testados.

- pipeline_modelo.py:
- Integra um pipeline completo que realiza o pré-processamento e a modelagem, com validação cruzada utilizando K-Fold.
- Implementa ajuste de hiperparâmetros com Optuna, otimizando os parâmetros de um GradientBoostingRegressor.
- Após a otimização, o pipeline é treinado e avaliado (tanto nos dados de treino quanto de teste), e os resultados são validados por meio de cross-validation.
- O pipeline final é salvo (usando joblib) para ser utilizado em produção.

- app.py:
- API construída com Flask para disponibilizar o modelo treinado.
- Carrega o pipeline salvo e oferece endpoints:
GET /: Rota inicial para verificação da disponibilidade da API.
POST /predict: Rota para receber dados em formato JSON, realizar a previsão e retornar a porcentagem e valor calculado da gorjeta.
GET /metrics: Exposição das métricas de requisições para monitoramento via Prometheus.
- Inclui logging estruturado para facilitar o rastreamento de erros e monitoramento de performance.

# Pré-requisitos
- Certifique-se de ter as seguintes dependências instaladas:

Python 3.8 ou superior
Bibliotecas Python:
pandas
numpy
scikit-learn
xgboost
optuna
joblib
flask
prometheus_client
seaborn
matplotlib
tqdm
Outros pacotes conforme necessidade

# Instalando as Dependências
- Utilize um ambiente virtual e instale as dependências via pip ou conda:
python -m venv venv
source venv/bin/activate  # Linux/macOS
venv\Scripts\activate     # Windows

pip install -r requirements.txt

# Estrutura de Pastas Sugerida
├── data
│   ├── raw
│   │   └── dataset_preprocessado.parquet
│   ├── processed
│   │   ├── dataset_processado.parquet
│   │   ├── feature_importances.csv
│   │   ├── feature_importances.parquet
│   │   └── train_test
│   │       ├── X_train.csv
│   │       ├── X_test.csv
│   │       ├── y_train.csv
│   │       └── y_test.csv
├── models
│   └── pipeline_model.pkl
├── reports
│   └── figures
│       ├── relacao_milhas_gorjeta.png
│       └── heatmap_correlacao.png
├── pipeline.py
├── treinando_modelos.py
├── modelo_previsao.py
├── pipeline_modelo.py
└── app.py


# Como Executar

1. Pré-processamento e Feature Engineering
Execute o script pipeline.py para:

Carregar os dados raw.
Realizar o pré-processamento e a criação de novas features.
Gerar visualizações e salvar os datasets processados.

python pipeline.py


2. Treinamento e Avaliação dos Modelos
Para treinar e comparar os modelos de regressão:
python treinando_modelos.py

Para executar o script de treinamento alternativo e avaliação:
python modelo_previsao.py

3. Otimização do Pipeline com Optuna
Utilize o script pipeline_modelo.py para:

Executar a otimização dos hiperparâmetros com Optuna.
Validar o pipeline com validação cruzada e salvar o pipeline treinado.
python pipeline_modelo.py

4. Deploy da API de Previsão
Inicie o servidor Flask para servir as previsões:
python app.py

- A API estará disponível em http://0.0.0.0:5000 e as métricas poderão ser acessadas via http://<host>:8000/metrics.

# Monitoramento e Logs
- Logging: Os scripts utilizam o módulo logging para registrar informações importantes e erros, facilitando a depuração.
- Métricas: A API implementa o monitoramento via Prometheus, permitindo a coleta de métricas de requisições e performance.

# Considerações Finais
Este projeto foi desenvolvido com foco em escalabilidade e eficiência, aplicando as melhores práticas de engenharia de dados e machine learning. Sinta-se à vontade para ajustar os pipelines, testar novas abordagens de modelagem e integrar novas técnicas de MLOps para aprimorar a performance e a robustez da solução.