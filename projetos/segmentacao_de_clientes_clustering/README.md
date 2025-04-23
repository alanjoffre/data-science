# Projeto de Segmentação de Clientes

Este projeto tem como objetivo segmentar clientes usando técnicas de machine learning, especificamente clustering com KMeans. O objetivo é agrupar clientes com características semelhantes para melhor análise e tomadas de decisão.

## Introdução

O projeto de Segmentação de Clientes visa identificar diferentes segmentos de clientes para ajudar em estratégias de marketing, desenvolvimento de produtos e melhorias no atendimento ao cliente. Utilizamos o algoritmo KMeans para criar clusters de clientes com base em dados de comportamento e demográficos.

## Estrutura do Projeto

A estrutura do projeto é a seguinte:

├── config │ └── config.yaml# Arquivo de configuração ├── data │ ├── raw # Dados brutos │ └── processed # Dados processados ├── logs # Arquivos de log ├── models # Modelos treinados ├── notebooks # Notebooks Jupyter ├── reports # Relatórios gerados ├── src │ ├── dashboard.py# Código do dashboard Streamlit │ └── main.py# Script principal para treinamento └── README.md# Documentação do projeto


## Ferramentas e Bibliotecas Utilizadas

Este projeto utiliza várias ferramentas e bibliotecas para análise de dados, machine learning e visualização. Aqui estão algumas das principais:

- **Python**: Linguagem de programação utilizada para todo o desenvolvimento do projeto.
- **Pandas**: Biblioteca para manipulação e análise de dados.
- **NumPy**: Biblioteca para operações matemáticas e array.
- **Scikit-learn**: Biblioteca para machine learning, utilizada para clustering com KMeans.
- **Matplotlib**: Biblioteca para criação de gráficos e visualizações.
- **Seaborn**: Biblioteca para visualização de dados baseada no Matplotlib.
- **Streamlit**: Biblioteca para criação de dashboards interativos.
- **PyYAML**: Biblioteca para leitura de arquivos YAML de configuração.
- **Joblib**: Biblioteca para salvar e carregar modelos treinados.
- **Logging**: Biblioteca padrão do Python para registro de logs de execução.
- **Spark**: Framework de processamento paralelo para análise de big data.
- **Kafka**: Plataforma de streaming distribuída utilizada para construir pipelines de dados em tempo real.

## Instalação

Para executar este projeto localmente, siga as instruções abaixo:

1. Clone o repositório:
   ```bash
   git clone https://github.com/seu-usuario/seu-repositorio.git

2. Navegue até o diretório do projeto:
cd seu-repositorio

3. Crie um ambiente virtual e ative-o:
python -m venv venv
source venv/bin/activate  # No Windows, use `venv\Scripts\activate`

4. Instale as dependências:
pip install -r requirements.txt

Uso
Para executar a aplicação de dashboard:

1. Execute o script do Streamlit:
streamlit run src/dashboard.py

Isso abrirá uma página web interativa com o dashboard, onde você poderá visualizar as análises e gráficos gerados a partir dos dados de segmentação de clientes.

Estrutura dos Dados
Os dados usados neste projeto são armazenados em um arquivo CSV com as seguintes colunas (traduzidas para português):

- Identificação
- Ano de Nascimento
- Escolaridade
- Estado Civil
- Renda
- Crianças em Casa
- Adolescentes em Casa
- Data de Cadastro
- Recência
- Gasto em Vinhos
- Gasto em Frutas
- Gasto em Produtos de Carne
- Gasto em Produtos de Peixe
- Gasto em Produtos Doces
- Gasto em Produtos de Ouro
- Compras em Promoções
- Compras pela Internet
- Compras por Catálogo
- Compras na Loja
- Visitas ao Site por Mês
- Aceitou Campanha 3
- Aceitou Campanha 4
- Aceitou Campanha 5
- Aceitou Campanha 1
- Aceitou Campanha 2
- Reclamação
- Custo de Contato
- Receita
- Resposta

## Configurações

As configurações do projeto estão armazenadas em um arquivo YAML (`config/config.yaml`). Aqui estão algumas das configurações importantes:

### Diretórios

- **data_path**: Caminho para o arquivo de dados CSV.
- **logs_path**: Caminho para os arquivos de log.
- **figures_path**: Caminho para salvar as figuras geradas.
- **reports_path**: Caminho para salvar os relatórios gerados.
- **models_path**: Caminho para salvar os modelos treinados.

### Pré-processamento

- **remove_outliers**:
  - `year_birth`: Ano de nascimento mínimo para exclusão de outliers.
  - `income`: Valor de renda máximo para exclusão de outliers.

### Modelo

- **params**:
  - `n_clusters`: Número de clusters para o KMeans.
  - `n_init`: Número de inicializações do KMeans.
  - `random_state`: Estado aleatório para reprodutibilidade.

### Exemplo de Arquivo de Configuração (`config.yaml`)

```yaml
data:
  source: 'data/raw/customer_segmentation.csv'
paths:
  logs: 'logs/'
  figures: 'figures/'
  reports: 'reports/'
  models: 'models/'
preprocessing:
  remove_outliers:
    year_birth: 1920
    income: 200000
model:
  params:
    n_clusters: 5
    n_init: 10
    random_state: 42
```
# Contato
Para mais informações, entre em contato com: Alan Joffre.
