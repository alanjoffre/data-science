# Projeto de Análise de Compartilhamento de Viagem - Classificação
- Este projeto tem como objetivo analisar dados de viagens e determinar se uma viagem compartilhada foi autorizada utilizando técnicas de classificação. Foram implementados diferentes modelos e pipelines para tratamento dos dados, avaliação e tuning dos hiperparâmetros, com destaque para o uso de Random Forest e calibradores, além de nested cross-validation e ajuste de threshold para otimização do F1-score.

# Sumário
- Descrição
- Arquitetura do Projeto
- Requisitos e Dependências
- Configuração do Ambiente
- Como Executar
- Treinamento e Avaliação de Múltiplos Modelos
- Pipeline Focado em Random Forest
- Estrutura dos Arquivos e Pastas
- Logs e Relatórios
- Considerações Finais

# Descrição
- O projeto aborda a análise de dados de compartilhamento de viagem utilizando algoritmos de classificação. Nele são executadas as seguintes etapas:

- Pré-processamento: Carregamento de dados a partir de arquivos Parquet e limpeza (imputação, remoção de outliers, feature engineering e padronização) previamente realizados.
- Divisão de Dados: Separação entre features e target, bem como divisão entre conjuntos de treino e teste.
- Modelagem: Treinamento de diversos modelos (Regressão Logística, Random Forest, Gradient Boosting, SVC, Extra Trees e XGBoost) com validação cruzada.
- Seleção de Modelo: Seleção do melhor modelo baseado no F1-score médio durante a validação cruzada.
- Avaliação Final: Métricas de acurácia, precisão, recall, F1-score, ROC AUC e geração de relatório de classificação.
- Random Forest com Tuning e Calibração: Pipeline específico que utiliza nested cross-validation para tuning de hiperparâmetros do Random Forest, calibrador (Platt Scaling) e ajuste de threshold para maximizar o F1-score. Geração de gráficos de ROC e Precision-Recall.

# Arquitetura do Projeto
O projeto está organizado em scripts e configurações que possibilitam a separação entre experimentos gerais e o pipeline específico de Random Forest:

- treinando_modelos.py:
Script responsável pelo treinamento e avaliação de múltiplos modelos de classificação. Realiza divisão dos dados, treinamento e validação cruzada e gera métricas finais no conjunto de teste.

- modelo_random_forest.py:
Script com foco na implementação de uma pipeline robusta com Random Forest. Inclui:

Carregamento e relatório exploratório dos dados.
Pré-processamento (escalonamento) e salvamento do pré-processador.
Nested Cross-Validation para tuning dos hiperparâmetros.
Calibração do modelo utilizando CalibratedClassifierCV.
Ajuste de threshold para maximização do F1-score.
Geração e salvamento dos gráficos ROC e Precision-Recall.
Persistência do modelo final e dos hiperparâmetros otimizados.
Requisitos e Dependências
Certifique-se de ter instaladas as seguintes dependências:

# Linguagem: Python 3.7+
Bibliotecas:
pandas
numpy
scikit-learn
xgboost
scipy
matplotlib
joblib
pyyaml
logging (módulo padrão do Python)

- Você pode instalar as dependências utilizando um arquivo requirements.txt. Exemplo:
pip install -r requirements.txt

Exemplo de conteúdo para o requirements.txt:

txt
Copiar
pandas
numpy
scikit-learn
xgboost
scipy
matplotlib
joblib
pyyaml

- Configuração do Ambiente

1. Arquivos de Configuração:

- O arquivo config/config.yaml deve conter os caminhos para logs, diretórios de modelos, métricas e dados processados.

- Certifique-se de que os diretórios definidos no YAML existam ou que o script os crie (no caso do modelo_random_forest.py, os diretórios são validados e criados automaticamente).

2. Dados:

O arquivo de dados processados (processado.parquet) deve estar disponível no caminho especificado no arquivo YAML e referenciado nos scripts.

# Como Executar
Treinamento e Avaliação de Múltiplos Modelos

1. Executar o script treinando_modelos.py:
python treinando_modelos.py

- Esse script irá:
Carregar os dados processados.
Dividir os dados em treino e teste.
Treinar diversos modelos (Logistic Regression, Random Forest, Gradient Boosting, SVC, Extra Trees e XGBoost).
Executar validação cruzada utilizando StratifiedKFold.
Selecionar o melhor modelo com base no F1-score médio.
Avaliar o melhor modelo no conjunto de teste e imprimir as métricas e o relatório de classificação.

# Pipeline Focado em Random Forest

1. Executar o script modelo_random_forest.py:
python modelo_random_forest.py

Esse script executa:

- Geração de um relatório inicial do dataset (número de linhas, tipos de dados, dados faltantes, duplicados e outliers).
- Divisão dos dados e aplicação de escalonamento via Pipeline.
- Salvamento do pré-processador (joblib).
- Nested cross-validation para tuning dos hiperparâmetros do Random Forest.
- Persistência dos melhores hiperparâmetros em um arquivo JSON.
- Calibração do modelo (usando Platt Scaling) e avaliação dos resultados.
- Ajuste do threshold para otimização do F1-score.
- Geração de gráficos de ROC e Precision-Recall, que são salvos nos diretórios configurados.
- Persistência do modelo final calibrado.

# Estrutura dos Arquivos e Pastas

- Uma estrutura sugerida para o projeto pode ser:

projeto_compartilhamento_viagem/
├── config/
│   └── config.yaml
├── data/
│   ├── raw/
│   └── processed/
│       └── processado.parquet
├── hyperparameters/
│   └── best_hyperparameters.json
├── logs/
│   └── modelo_random_forest.log
├── models/
│   └── final_model.joblib
├── reports/
│   ├── figures/
│   │   ├── roc_curve.png
│   │   └── precision_recall_curve.png
│   └── metrics.json
├── treinando_modelos.py
├── modelo_random_forest.py
├── requirements.txt
└── README.md

- Observação: Os caminhos absolutos presentes nos scripts podem ser adaptados para caminhos relativos, dependendo da estrutura do seu ambiente de desenvolvimento.

# Logs e Relatórios

- Logs:
Os scripts configuram logs (por exemplo, modelo_random_forest.log) para registrar informações importantes durante a execução, como número de linhas importadas, hiperparâmetros encontrados e caminhos dos gráficos gerados.

- Relatórios e Métricas:
As métricas (acurácia, precisão, recall, F1-score e Nested CV F1-score) são salvas em um arquivo JSON. Os gráficos ROC e Precision-Recall também são gerados e salvos para análises futuras.

# Considerações Finais

- Boas Práticas:

- A utilização de pipelines para pré-processamento garante a reprodutibilidade e consistência do fluxo de dados.
- O uso de nested cross-validation para tuning de hiperparâmetros evita o viés de seleção e fornece uma estimativa mais robusta da performance.
- A calibração do modelo e o ajuste de threshold demonstram o foco na otimização do F1-score, essencial em problemas de classificação com classes desbalanceadas.

- Escalabilidade e Inovação:
Este projeto pode ser expandido com a integração de novas técnicas de feature engineering, frameworks de MLOps para automação de pipelines e monitoramento de modelos em produção, mantendo o foco em performance e escalabilidade.