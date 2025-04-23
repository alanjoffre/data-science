# Projeto de Análise de Sentimentos

Este projeto visa realizar uma análise de sentimentos em um conjunto de dados de grande porte, utilizando a biblioteca Dask para manipulação de dados em paralelo e outras técnicas avançadas de processamento de linguagem natural (NLP).

## Estrutura do Projeto

O projeto é organizado em várias etapas, cada uma contendo scripts específicos para execução e testes unitários:

1. **Carregamento de Dados**: Leitura e carregamento do conjunto de dados bruto.
2. **Pré-processamento de Dados**: Limpeza e preparação dos dados para análise.
3. **Análise Exploratória de Dados (EDA)**: Exploração visual e estatística dos dados.
4. **Preparação de Dados para Modelagem**: Transformação e divisão dos dados em conjuntos de treino e teste.
5. **Modelagem de Machine Learning**: Treinamento de modelos de aprendizado de máquina para classificação de sentimentos.
6. **Avaliação do Modelo**: Avaliação dos modelos utilizando métricas de desempenho.
7. **Implementação e Monitoramento**: Implementação dos modelos em produção e monitoramento do desempenho.

## Bibliotecas e Ferramentas Utilizadas

- **Manipulação e Análise de Dados**:
  - `numpy`
  - `pandas`
  - `dask`
  - `pyarrow`

- **Processamento de Linguagem Natural (NLP)**:
  - `nltk`
  - `gensim`
  - `spacy`
  - `vaderSentiment`
  - `textblob`
  - `symspellpy`

- **Modelagem e Machine Learning**:
  - `scikit-learn`
  - `imbalanced-learn`
  - `xgboost`
  - `lightgbm`
  - `catboost`
  - `tensorflow`
  - `transformers`
  - `torch`
  - `keras`

- **Utilitários e Outros**:
  - `joblib`
  - `pytest`
  - `cython`
  - `pandas-profiling`
  - `matplotlib`
  - `seaborn`
  - `wordcloud`
  - `Pillow`
  - `flask`
  - `dask-ml`
  - `typing-extensions`
  - `json`
  - `yaml`
  - `logging`
  - `time`
  - `os`
  - `dateutil`

## Tamanho do Dataset
- Acima de 10.000.000 linhas

## Processamento em Bloco
Utilizamos processamento em bloco para lidar eficientemente com o grande volume de dados, garantindo que as operações sejam realizadas de maneira paralela e eficiente.

## Instalação

1. Clone o repositório:
    ```bash
    git clone https://github.com/seu-usuario/analise-de-sentimentos.git
    cd analise-de-sentimentos
    ```

2. Crie um ambiente virtual e ative-o:
    ```bash
    python -m venv venv
    source venv/bin/activate  # No Windows, use: venv\Scripts\activate
    ```

3. Instale as dependências:
    ```bash
    pip install -r requirements.txt
    ```

## Uso

### 1. Carregamento e Pré-processamento de Dados
Utilize os scripts da pasta `src` para carregar e pré-processar os dados.

### 2. Treinamento de Modelos
Treine os modelos de classificação de sentimentos utilizando os scripts disponíveis.

### 3. Avaliação e Otimização
Avalie o desempenho dos modelos e utilize a busca de hiperparâmetros para otimização.

## Contribuição
Contribuições são bem-vindas! Sinta-se à vontade para abrir issues ou enviar pull requests.

## Licença
Este projeto está licenciado sob a Licença MIT.

## Autor
- [Alan Joffre](https://github.com/alanjoffre/data-science/tree/master/projetos/analise-de-sentimentos)
