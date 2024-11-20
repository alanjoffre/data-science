# Projeto de Análise de Sentimentos

Este projeto tem como objetivo realizar uma análise de sentimentos em um conjunto de dados grande, utilizando a biblioteca Dask para manipulação de dados em paralelo.

## Estrutura do Projeto

```plaintext
analise-de-sentimentos/
├── config/
│   └── config.yaml
├── data/
│   ├── processed/
│   └── raw/
│       └── asentimentos.parquet
├── logs/
├── models/
├── predictions/
├── reports/
│   ├── figures/
│   └── README.md
├── src/
│   ├── etapa1_carregamento_dados.py
│   ├── etapa1_teste_unitario.py
│   ├── etapa2_preprocessamento.py
│   ├── etapa2_teste_unitario.py
│   ├── etapa3_analise_exploratoria.py
│   ├── etapa3_teste_unitario.py
│   ├── etapa4_preparacao_dados.py
│   ├── etapa4_teste_unitario.py
│   ├── etapa5_modelagem.py
│   ├── etapa5_teste_unitario.py
│   ├── etapa6_avaliacao_modelo.py
│   ├── etapa6_teste_unitario.py
│   ├── etapa7_implementacao_monitoramento.py
│   ├── etapa7_teste_unitario.py
│   └── etapa8_dashboard.py
└── README.md
```
# Instruções de Configuração
- Crie um ambiente virtual:

No Windows:
python -m venv venv
venv\Scripts\activate

# Instale as dependências:
pip install -r requirements.txt

# Baixar e Instalar o Modelo en_core_web_sm do spaCy:
python -m spacy download en_core_web_sm

# Executando o Projeto
## Etapa 1: Carregamento de Dados

- Execute o script de carregamento de dados:
python src/etapa1_carregamento_dados.py

- Teste a Etapa de Carregamento de Dados:
python src/etapa1_teste_unitario.py

# Estrutura do Projeto
O projeto é organizado em várias etapas, cada uma contendo scripts específicos para execução e testes unitários:

- Etapa 1: Carregamento de Dados
- Etapa 2: Pré-processamento de Dados
- Etapa 3: Análise Exploratória de Dados (EDA)
- Etapa 4: Preparação de Dados para Modelagem
- Etapa 5: Modelagem de Machine Learning
- Etapa 6: Avaliação do Modelo
- Etapa 7: Implementação e Monitoramento
- Etapa 8: Dashboard Completo com Flask

# Diretórios e Arquivos
- config/: Contém o arquivo de configuração do projeto config.yaml.
- data/raw/: Contém o dataset bruto asentimentos.parquet.
- data/processed/: Armazena os datasets processados de cada etapa.
- logs/: Armazena os arquivos de log das execuções dos scripts.
- models/: Salva os modelos treinados.
- predictions/: Armazena as previsões geradas pelos modelos.
- reports/figures/: Contém gráficos e visualizações gerados durante o projeto.
- src/: Contém os scripts das etapas e os testes unitários.

# Contato
- Para mais informações ou dúvidas, entre em contato.