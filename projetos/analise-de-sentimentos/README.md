# Projeto: Analise de Sentimentos X

- O arquivo README.md incluirá as informações do projeto e instruções para executar o código.

# Projeto de Análise de Sentimentos

## Descrição
Este projeto visa realizar uma análise de sentimentos em um grande conjunto de dados utilizando as bibliotecas Dask e Spacy para lidar com a performance e processamento de dados.

## Estrutura de Diretórios
```plaintext
D:\Github\data-science\projetos\analise-de-sentimentos\nlp\
│
├── config\
│   └── config.yaml
├── data\
│   ├── raw\
│   │   └── asentimentos.parquet
│   ├── processed\
│   └── interim\
├── logs\
├── models\
├── predictions\
├── reports\
│   ├── figures\
│   └── EDA_report.html
└── src\
    ├── etl\
    │   └── load_data.py
    ├── preprocess\
    │   └── data_cleaning.py
    │   └── text_normalization.py
    │   └── tokenization.py
    │   └── remove_stopwords.py
    ├── eda\
    ├── modeling\
    ├── evaluation\
    ├── deployment\
    └── utils\
```
# Dependências
- Dask
- Pandas
- Scikit-learn
- Matplotlib
- Seaborn
- Spacy
- Cython
- Visions
- Pandas-profiling
- Pydantic
- PyYAML
- Fastparquet
- Pyarrow

# Como executar
- 1 - Configure o ambiente virtual:
python -m venv venv_nome
.\venv_nome\Scripts\activate  # Windows

- 2 - Instale as dependências:
pip install -r requirements.txt

- 3 - Instale o modelo spaCy en_core_web_sm:
python -m spacy download en_core_web_sm

- 4 - Execute o script de limpeza de dados:
python src/etl/load_data.py

- 5 - Execute o script de limpeza de dados:
python src/preprocess/data_cleaning.py

- 6 - Execute o script de normalização de texto:
python src/preprocess/text_normalization.py

- 7 - Execute o script de tokenização de texto:
python src/preprocess/tokenization.py

- 8 - Execute o script de remoção de stop words:
python src/preprocess/remove_stopwords.py

- 9 - Execute o script de stemming e lematização:
python src/preprocess/stemming_lemmatization.py

# Contato: 
## https://alanjoffre.github.io/my-profile/

