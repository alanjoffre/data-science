# Análise de Desempenho de Viagens Agrupadas com Clusterização de Séries Temporais.
- Este projeto implementa um pipeline completo para análise de desempenho de viagens agrupadas, utilizando técnicas avançadas de pré-processamento, engenharia de features, segmentação com janelas sobrepostas, normalização e clusterização de séries temporais. O pipeline também inclui a otimização dos hiperparâmetros via Optuna e a comparação com modelos alternativos (K-Shape e DBSCAN), registrando todos os resultados e parâmetros no MLflow.

# Sumário
- Descrição
- Motivação
- Tecnologias Utilizadas
- Arquitetura do Projeto
- Instalação e Configuração
- Execução
- Resultados e Interpretação
- Melhorias Futuras
- Contribuição
- Licença

# Descrição
- ste projeto tem como objetivo analisar o desempenho de viagens a partir de dados históricos, utilizando clusterização de séries temporais. O pipeline realiza as seguintes etapas:

1. Ingestão de Dados: Carregamento da configuração (arquivo YAML) e do dataset (formato Parquet).
2. Pré-processamento e Engenharia de Features:
Conversão da coluna de data para o formato datetime e agregação diária (média da tarifa).
Cálculo de features adicionais, como média móvel, desvio padrão móvel, componentes de tendência e sazonalidade via decomposição STL (com período fixado em 7) e quantis (25º e 75º percentil).
3. Segmentação com Janelas Sobrepostas:
Os dados são segmentados em janelas fixas (ex.: 30 dias) com sobreposição (definida pelo parâmetro step), permitindo a captura de padrões em múltiplas escalas temporais.
4. Normalização:
Aplicação do StandardScaler para padronizar as features, melhorando a qualidade da clusterização.
5. Otimização de Hiperparâmetros:
Uso do Optuna para otimizar o número de clusters, o tamanho da janela e o passo, maximizando o Silhouette Score.
6. Clusterização:
Aplicação do TimeSeriesKMeans com a métrica DTW usando os melhores parâmetros otimizados.
7. Modelos Alternativos:
Execução de modelos alternativos de clusterização, como K-Shape e DBSCAN, para comparação de métricas.
8. Registro e Monitoramento:
Todos os parâmetros, métricas e resultados são registrados no MLflow, permitindo rastreamento e reprodutibilidade dos experimentos.

# Motivação
- Identificar Padrões de Desempenho:
A análise de séries temporais permite detectar períodos com comportamentos distintos nas viagens (por exemplo, alta volatilidade versus estabilidade).
- Tomada de Decisão Baseada em Dados:
Ao extrair e agrupar padrões, o projeto fornece insights que podem embasar decisões estratégicas e operacionais.
- Otimização e Validação:
A utilização de técnicas de otimização (Optuna) e a comparação com modelos alternativos garantem que o pipeline está ajustado para extrair o máximo de informação dos dados.

# Tecnologias Utilizadas
- Python 3.8
- Bibliotecas Principais:
Pandas, NumPy: Manipulação e agregação de dados.
Matplotlib: Visualização dos resultados.
tslearn: Clusterização de séries temporais (TimeSeriesKMeans, K-Shape).
Scikit-learn: Cálculo de métricas (Silhouette Score) e normalização (StandardScaler).
DBSCAN: Clusterização alternativa aplicada aos dados flatten.
MLflow: Registro e monitoramento dos experimentos.
Optuna: Otimização de hiperparâmetros via Bayesian Optimization.
Statsmodels: Decomposição STL para extração de tendência e sazonalidade.
YAML: Leitura do arquivo de configuração.

# Arquitetura do Projeto
- O pipeline está organizado de forma modular, com as seguintes etapas:

1. Ingestão e Configuração:
Funções load_config e load_dataset para ler arquivos de configuração e dados.
2. Pré-processamento e Engenharia de Features:
Função preprocess_data que agrega os dados por data e calcula diversas features (média, média móvel, desvio padrão, trend, seasonal, quantis).
3. Segmentação e Normalização:
Função segment_time_series para criar janelas sobrepostas e normalize_segments para padronizar os dados.
4. Otimização de Hiperparâmetros:
Função objective e optimize_hyperparameters utilizando Optuna para encontrar os melhores parâmetros.
5. Clusterização:
Aplicação do modelo principal (TimeSeriesKMeans) e modelos alternativos (K-Shape e DBSCAN).
6. Registro e Visualização:
Registro dos experimentos com MLflow e visualização dos centróides dos clusters.

# Instalação e Configuração
1. Clone o Repositório:
git clone https://github.com/seu_usuario/seu_projeto.git
cd seu_projeto

2. Crie um Ambiente Virtual e Instale as Dependências:
python -m venv venv
source venv/bin/activate   # No Windows: venv\Scripts\activate
pip install -r requirements.txt

3. Configuração:
Verifique e ajuste os caminhos no arquivo de configuração YAML (config/config.yaml), que contém os diretórios e parâmetros do projeto.

# Execução
- Para executar o pipeline, basta rodar:
python pipeline.py

- Durante a execução, o pipeline realizará as seguintes etapas:
Carregamento da configuração e dos dados.
Pré-processamento e extração de features (incluindo decomposição STL e cálculo de quantis).
Segmentação dos dados com janelas sobrepostas e normalização.
Otimização dos hiperparâmetros via Optuna para maximizar o Silhouette Score.
Clusterização final com TimeSeriesKMeans e execução de modelos alternativos (K-Shape e DBSCAN).
Registro dos parâmetros, métricas e resultados com MLflow.
Visualização dos centróides dos clusters.

# Resultados e Interpretação
- Melhores Hiperparâmetros:
A otimização via Optuna selecionou os melhores parâmetros (por exemplo, window_size, step e número de clusters) que maximizam o Silhouette Score.
- Modelo Principal:
O TimeSeriesKMeans, configurado com os melhores parâmetros, obteve um Silhouette Score alto (por exemplo, ~0.62), indicando uma boa separação dos clusters.
- Modelos Alternativos:
Os resultados dos modelos K-Shape e DBSCAN foram comparados, e suas métricas foram registradas para análise.
- Visualização:
Os centróides dos clusters são plotados para facilitar a interpretação dos padrões identificados.

Esses resultados permitem extrair insights valiosos sobre a dinâmica do desempenho das viagens e identificar períodos com comportamentos distintos.

# Melhorias Futuras
- Novas Features:
Explorar indicadores adicionais (autocorrelação, volatilidade, features externas como condições climáticas ou feriados).
- Segmentação Multi-escala:
Implementar janelas dinâmicas ou segmentação em múltiplas escalas para capturar padrões de curto e longo prazo.
- Modelos de Clusterização Alternativos:
Experimentar outros algoritmos (por exemplo, HDBSCAN, autoencoders para embeddings) e realizar ensemble clustering.
- Otimização Avançada:
Expandir o espaço de busca do Optuna e testar outros métodos de otimização.
- Integração e Deploy:
Containerizar o projeto com Docker e orquestrar com Kubernetes para deploy em produção.

# Contribuição
- Contribuições são bem-vindas! Se você deseja contribuir:

Faça um fork do repositório.
Crie uma branch para sua feature (git checkout -b feature/nova-feature).
Faça as alterações e commit (git commit -am 'Adiciona nova feature').
Envie sua branch (git push origin feature/nova-feature).
Abra um Pull Request com uma descrição detalhada das alterações.

# Licença
Veja o arquivo LICENSE para mais detalhes.