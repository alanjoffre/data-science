# Projeto de Detecção de Anomalias em Tarifas de Compartilhamento de Viagem.
- Este repositório contém um projeto completo para análise e detecção de anomalias em tarifas de viagens compartilhadas. O projeto aplica técnicas avançadas de machine learning e engenharia de dados, integrando diferentes modelos (supervisionados e não supervisionados) e pipelines de pré-processamento para identificar comportamentos atípicos em dados logísticos e de transporte.

# Sumário
- Visão Geral
- Arquitetura do Projeto
- Pré-requisitos e Dependências
- Configuração e Estrutura de Diretórios
- Pipeline de Processamento e Modelagem
- Carregamento e Configuração
- Tratamento e Limpeza dos Dados
- Pré-processamento e Engenharia de Features
- Treinamento e Otimização dos Modelos
- Execução e Geração de Artefatos
- Métricas e Validação
- Considerações Finais

# Visão Geral
- O projeto tem como objetivo desenvolver um pipeline robusto para a detecção de anomalias em tarifas, utilizando tanto modelos supervisionados (Random Forest) quanto não supervisionados (Isolation Forest, Local Outlier Factor, Autoencoder). A estratégia de modelagem inclui refinamento progressivo dos hiperparâmetros (através de múltiplos GridSearchCV) e validação cruzada para garantir a reprodutibilidade e robustez dos resultados.

# Arquitetura do Projeto
A estrutura do repositório está organizada para favorecer escalabilidade e manutenibilidade. Entre os principais componentes, destacam-se:

- Configuração: Arquivo config.yaml com definição dos paths para dados, modelos, logs e relatórios.
- Pré-processamento e Tratamento: Scripts dedicados à limpeza, transformação e engenharia de features, com logs detalhados.
- Modelagem: Scripts para treinamento e avaliação dos modelos, incluindo otimização de hiperparâmetros em múltiplos níveis (grid, refinado, fino e ultra fino).
- Automação e Registro: Pipeline estruturado para salvar artefatos (modelos, escaladores, métricas) e monitorar o desempenho por meio de logs.

# Pré-requisitos e Dependências
O projeto foi desenvolvido utilizando Python 3.8 e diversas bibliotecas modernas para ciência de dados e machine learning. Confira os requisitos no arquivo requirements.txt. Entre as dependências destacam-se:

- Pandas e NumPy para manipulação e análise de dados;
- Scikit-learn para modelagem, pré-processamento e avaliação;
- TensorFlow/Keras para construção do autoencoder;
- Joblib para serialização de modelos e escaladores;
- Matplotlib e Seaborn para geração de visualizações;
- PyYAML para gerenciamento das configurações do projeto.

# Configuração e Estrutura de Diretórios
O arquivo config.yaml define os caminhos para:

- config_dir: Diretório com arquivos de configuração.
- logs_dir: Diretório para logs detalhados do pipeline.
- notebook_dir: Para notebooks exploratórios.
- src_dir: Código-fonte do projeto.
- models: Local para salvar modelos treinados e escaladores.
- data: Armazenamento dos dados brutos e processados.
- reports: Relatórios e figuras geradas durante a análise.
- metrics: Arquivos de métricas para avaliação dos modelos.
Essa organização garante que os artefatos do projeto sejam facilmente acessíveis e integrados aos processos de CI/CD e MLOps.

# Pipeline de Processamento e Modelagem
Carregamento e Configuração
- O script inicia lendo o arquivo config.yaml, definindo paths para dados, modelos e logs.
- São criados os diretórios necessários para salvar os artefatos e garantir que o ambiente esteja corretamente estruturado.

# Tratamento e Limpeza dos Dados
- Verificação de Duplicidade: Identificação e remoção de registros duplicados.
- Imputação de Valores Faltantes: Aplicação de estratégias de imputação (mediana para variáveis numéricas e moda para categóricas).
- Correção de Limites: Validação de valores fora dos limites esperados (clip) para variáveis numéricas.
- Detecção e Correção de Outliers: Uso do método IQR para identificar outliers e aplicar winsorização.

# Pré-processamento e Engenharia de Features
- Conversão de colunas de datas e engenharia de features temporais (hora, dia, mês, dia da semana).
- Conversão de variáveis categóricas (ex.: setores censitários) para garantir tratamento adequado nas análises.
- Escalonamento dos dados utilizando StandardScaler para normalização.

# Treinamento e Otimização dos Modelos
- Modelos Supervisionados: Utilização do Random Forest com múltiplos estágios de refinamento de hiperparâmetros utilizando GridSearchCV.
- Modelos Não Supervisionados: Implementação de Isolation Forest, Local Outlier Factor e Autoencoder para detecção de anomalias.
- Cada modelo é avaliado com base na métrica F1 (com a classe de anomalias definida como -1) e os melhores parâmetros são selecionados por meio de validação cruzada.

# Execução e Geração de Artefatos
- Serialização dos Artefatos: Após o treinamento, os modelos, escaladores e pré-processadores são salvos em formato .joblib, facilitando a reutilização em produção.
- Métricas: As métricas obtidas em cada estágio (modelo padrão, grid search, refinamento fino, ultra fino) são salvas em formato JSON para rastreabilidade e comparação.
- Relatórios e Visualizações: São gerados relatórios com estatísticas descritivas e gráficos para análise exploratória dos dados, além de logs detalhados das operações realizadas.

# Métricas e Validação
A validação dos modelos é realizada utilizando técnicas de:

- Train/Test Split com estratificação para garantir a representatividade da classe de anomalias.
- Validação Cruzada: Aplicada tanto nos dados de treinamento quanto nos dados de teste para confirmar a robustez do modelo.
- Métricas: Foco na métrica F1-score, configurada para tratar anomalias (classe -1) como positiva.
A comparação entre modelos permite identificar o melhor método para o cenário proposto, garantindo que o pipeline possa ser adaptado conforme a evolução dos dados.

# Considerações Finais
Este projeto foi desenvolvido com foco em escalabilidade, robustez e reprodutibilidade. O uso de práticas avançadas de engenharia de dados, aliado a técnicas de modelagem de ponta e uma estrutura de pipeline modular, permite a rápida adaptação a novos cenários e requisitos.

Para maiores informações e contribuições, consulte a documentação interna e os comentários presentes no código-fonte.

# Como Executar o Projeto:

1. Clone o repositório:
git clone https://github.com/seu-usuario/seu-projeto.git
cd seu-projeto

2. Instale as dependências:
pip install -r requirements.txt

3. Configure o arquivo config.yaml com os paths correspondentes ao seu ambiente.

4. Execute os scripts de pré-processamento e treinamento:
python src/Modelos.py
python src/Modelo.py
python src/tratamento_limpeza.py

5. Verifique os artefatos gerados (modelos, logs, métricas e relatórios) nos diretórios configurados.
python src/Modelos.py




