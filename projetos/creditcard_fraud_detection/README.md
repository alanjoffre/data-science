# Credit Card Fraud Detection

Este projeto tem como objetivo detectar fraudes em transações de cartões de crédito usando técnicas de aprendizado de máquina. Utilizamos um modelo de Random Forest aprimorado para identificar transações fraudulentas com base em dados de transações anteriores.

## Índice
1. [Descrição do Projeto](#descrição-do-projeto)
2. [Estrutura do Projeto](#estrutura-do-projeto)
3. [Instalação e Configuração](#instalação-e-configuração)
4. [Uso](#uso)
5. [Arquivos Principais](#arquivos-principais)
6. [Análise Segmentada](#análise-segmentada)
7. [Resultados](#resultados)
8. [Contribuições](#contribuições)
9. [Licença](#licença)

## Descrição do Projeto
O projeto utiliza dados históricos de transações de cartões de crédito para treinar um modelo de Random Forest capaz de detectar fraudes. O modelo é treinado, ajustado e avaliado com uma série de técnicas para garantir a máxima precisão e desempenho. Além disso, realizamos uma análise segmentada para avaliar o desempenho do modelo em diferentes categorias de valor de transação.

## Estrutura do Projeto
creditcard_fraud_detection/ │ ├── config/ │ └── config.yaml│ ├── data/ │ ├── processed_data/ │ │ └── creditcard_processed.csv │ └── real_data/ │ └── creditcard_real.csv │ ├── models/ │ └── best_rf_model_with_threshold.pkl │ ├── preprocessors/ │ └── smote.pkl│ ├── reports/ │ └── figures/ │ └── proporcao_fraudes_categoria_valor_transacao.png │ ├── src/ │ ├── app.py│ ├── consumer.py│ ├── producer.py│ ├── modelagem_analise_segmentada.py │ └── creditcard/ │ └── README.md


## Instalação e Configuração
1. Clone o repositório:
    ```bash
    git clone https://github.com/seu_usuario/creditcard_fraud_detection.git
    cd creditcard_fraud_detection
    ```

2. Crie um ambiente virtual e ative-o:
    ```bash
    python -m venv creditcard
    source creditcard/bin/activate # Linux/Mac
    creditcard\Scripts\activate # Windows
    ```

3. Instale as dependências:
    ```bash
    pip install -r requirements.txt
    ```

4. Configure o ambiente:
    - Crie os diretórios necessários: `models`, `preprocessors`, e `reports/figures`.
    - Atualize o arquivo `config.yaml` com os caminhos corretos.

## Uso
### 1. Preparação do Ambiente
Inicie o Zookeeper e o Kafka:
cd C:\kafka_2.13-3.9.0
.\bin\windows\zookeeper-server-start.bat .\config\zookeeper.properties
.\bin\windows\kafka-server-start.bat .\config\server.properties

2. Produção de Dados
Produza os dados de transações:
cd creditcard/src
python producer.py

3. Consumo de Dados e Previsões
Consuma os dados e faça previsões:
python consumer.py

4. API Flask
Inicie o servidor Flask:
python app.py

## Arquivos Principais
- modelagem_analise_segmentada.py: Script para treinar o modelo de Random Forest, ajustar hiperparâmetros, realizar validação cruzada e análise segmentada.

- app.py: API Flask para servir previsões de transações.

- consumer.py: Consumidor Kafka que envia transações para a API Flask e recebe previsões.

- producer.py: Produtor Kafka que envia dados de transações reais para o tópico Kafka.

# Análise Segmentada
O script modelagem_analise_segmentada.py também realiza uma análise segmentada, ajustando o threshold para diferentes categorias de valor de transação e avaliando o desempenho do modelo em cada segmento.

# Resultados
Os resultados do modelo treinado, incluindo gráficos e relatórios de classificação, são salvos no diretório reports/figures. A análise segmentada mostrou que o modelo tem um desempenho particularmente bom na categoria de transações de valor 0-50.

# Contribuições
Contribuições são bem-vindas! Sinta-se à vontade para abrir issues ou pull requests no repositório GitHub.

# Licença
Este projeto está licenciado sob a Licença MIT. Consulte o arquivo LICENSE para obter mais informações.