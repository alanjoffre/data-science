# Previsão de Gorjetas – Pipeline e Modelagem de Dados.
 - Este repositório contém um conjunto de scripts e pipelines para o processamento de dados, treinamento, avaliação e deploy de modelos de machine learning aplicados à previsão de gorjetas em operações de logística e transporte. Os processos englobam desde a preparação dos dados até a disponibilização de uma API em Flask para servir as previsões, com monitoramento via Prometheus.

 ## Visão Geral do Projeto
- O projeto foi desenvolvido utilizando Python e diversas bibliotecas de data science e machine learning (como scikit-learn, XGBoost, Optuna, entre outras), com o objetivo de criar soluções escaláveis e de alta performance para previsão de gorjetas. A estrutura do projeto contempla:

- Pré-processamento e Engenharia de Features: Conversão e tratamento de colunas (datas, numéricas e categóricas), criação de novas variáveis e amostragem de dados.

- Treinamento de Modelos: Implementação e avaliação de múltiplos algoritmos (Regressão Linear, Random Forest, Gradient Boosting, KNeighbors, SVR e XGBoost) com técnicas de ajuste de hiperparâmetros (Grid Search e Random Search).

- Otimização de Pipeline com Optuna: Pipeline otimizado que integra pré-processamento e modelagem, validado via validação cruzada.

- Deploy e API: Uma aplicação Flask que carrega o pipeline treinado, disponibilizando uma rota para previsões e outra para métricas expostas ao Prometheus.

