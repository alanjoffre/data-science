# Projeto de Predição de Churn de Clientes

Este projeto tem como objetivo construir um modelo preditivo para identificar clientes com maior risco de cancelamento (*churn*), utilizando dados históricos de clientes. A previsão de *churn* ajuda a empresa a tomar ações proativas para reter clientes e melhorar a experiência do usuário.

## Objetivos

- Prever quais clientes têm maior probabilidade de deixar a empresa (*churn*).
- Analisar e tratar os dados de clientes para otimizar o modelo preditivo.
- Construir e treinar modelos de machine learning, incluindo Random Forest e XGBoost.
- Avaliar a performance dos modelos utilizando métricas como acurácia, precisão, recall, e curva ROC.

## Estrutura do Projeto

```plaintext
rotatividade-de-clientes/
├── machine-learning/
│   ├── config/
│   │   └── config.yaml                # Configurações do projeto
│   ├── data/
│   │   ├── raw/
│   │   │   └── rclientes.csv          # Dados brutos
│   │   └── processed/
│   │       └── rclientes_preprocessado.csv  # Dados processados
│   ├── models/
│   │   └── xgboost_model.joblib      # Modelo treinado
│   ├── notebooks/
│   │   └── churn_prediction.ipynb    # Notebook de análise e modelagem
│   ├── reports/
│   │   ├── classification_report_1.txt  # Relatório de classificação 1
│   │   ├── classification_report_2.txt  # Relatório de classificação 2
│   │   └── classification_report_3.txt  # Relatório de classificação 3
│   └── src/
│       ├── data_preprocessing.py      # Script de pré-processamento dos dados
│       ├── model_training.py          # Script de treinamento de modelos
│       └── evaluation.py              # Avaliação do modelo
├── requirements.txt                  # Dependências do projeto
└── README.md                         # Este arquivo
```

## Tecnologias Utilizadas
- Python: Linguagem de programação utilizada no desenvolvimento.
- Pandas: Manipulação e análise de dados.
- NumPy: Operações numéricas.
- Scikit-learn: Modelagem preditiva e avaliação de modelos.
- XGBoost: Modelo de machine learning para classificação.
- Matplotlib & Seaborn: Visualizações de dados e análise exploratória.
- Imbalanced-learn: Técnicas para lidar com dados desbalanceados.

## Como Executar o Projeto

# 1. Clone o repositório
git clone https://github.com/seu-usuario/rotatividade-de-clientes.git
cd rotatividade-de-clientes

# 2. Instale as dependências
- É recomendado usar um ambiente virtual. Você pode instalar as dependências com o seguinte
pip install -r requirements.txt

# 3. Prepare os Dados
- Os dados brutos são encontrados na pasta data/raw/. Antes de treinar o modelo, os dados devem ser processados através do script de pré-processamento:
python src/data_preprocessing.py

- Isso gera os dados processados, que serão usados para o treinamento do modelo.

# 4. Treine o Modelo
- Para treinar o modelo de previsão de churn, execute o seguinte script:
python src/model_training.py

# 5. Avalie o Modelo
- Após o treinamento, os modelos serão avaliados e os relatórios de classificação serão gerados na pasta reports/. Para avaliar o modelo, execute:
python src/evaluation.py

- Os relatórios de classificação serão gerados nas pastas apropriadas.

# Como Utilizar os Modelos
- O modelo treinado será salvo na pasta models/ como xgboost_model.joblib. Para utilizá-lo em um novo conjunto de dados, carregue o modelo da seguinte forma:

import joblib

# Carregar o modelo treinado
model = joblib.load('models/xgboost_model.joblib')

# Previsões com o modelo carregado
predictions = model.predict(X_new_data)

# Estrutura de Pastas
- machine-learning/config: Contém o arquivo de configuração config.yaml que armazena caminhos e parâmetros do projeto.
- machine-learning/data: Contém os dados brutos e processados. O arquivo rclientes.csv é o conjunto de dados bruto, enquanto o rclientes_preprocessado.csv contém os dados prontos para modelagem.
- machine-learning/models: Contém os modelos treinados, incluindo o modelo XGBoost salvo.
- machine-learning/notebooks: Contém os notebooks usados para análise e experimentação.
- machine-learning/reports: Contém os relatórios de avaliação do modelo.
- machine-learning/src: Contém os scripts principais, como pré-processamento, treinamento de modelos e avaliação.

# Dependências
- Crie um ambiente virtual e instale as dependências com o comando:
pip install -r requirements.txt

# Arquivo requirements.txt:
pandas
numpy
scikit-learn
xgboost
imbalanced-learn
matplotlib
seaborn
joblib

# Licença
- Este projeto está licenciado sob a Licença MIT - veja o arquivo LICENSE para mais detalhes.
