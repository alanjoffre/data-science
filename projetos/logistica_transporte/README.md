# Projeto: Logística e Transporte - 13 cenários aplicando modelos de Machine Learning e Redes Neurais.

## Descrição do Projeto

Este projeto tem como objetivo analisar dados de viagens de uma empresa de transporte e logística para otimizar operações e melhorar a eficiência. Utilizando técnicas de Machine Learning e análises de dados, buscamos insights valiosos para diversas áreas, como previsão de gorjeta, comportamento de compartilhamento de viagem, desempenho de viagens agrupadas, detecção de anomalias em tarifas, análise geoespacial, modelagem de receita, eficácia das cobranças adicionais, otimização de rotas, gerenciamento de frota, análise de demanda, eficiência operacional, planejamento urbano e análise de custos.

## Estrutura do Projeto

1. **Previsão de Gorjeta (Regressão)**
   - Objetivo: Prever o valor da gorjeta com base em várias características da viagem.
   - Métodos: Regressão Linear, Random Forest Regressor, Gradient Boosting Regressor.
   - Resultado Esperado: Um modelo que pode prever com precisão o valor da gorjeta esperado para uma viagem, considerando fatores como a distância da viagem, a área de origem e destino, e o total da viagem.

2. **Análise de Comportamento de Compartilhamento de Viagem (Classificação)**
   - Objetivo: Analisar os fatores que influenciam a autorização de viagens compartilhadas.
   - Métodos: Decision Trees, Random Forest, Logistic Regression.
   - Resultado Esperado: Um modelo que pode prever se uma viagem será compartilhada com base em características como a distância da viagem, a área de origem e destino, e o total da viagem.

3. **Análise de Desempenho de Viagens Agrupadas (Clusterização e Séries Temporais)**
   - Objetivo: Avaliar a eficiência e o impacto das viagens agrupadas.
   - Métodos: K-means para clusterização de viagens semelhantes, Modelos de séries temporais.
   - Resultado Esperado: Insights sobre a eficiência das viagens agrupadas e identificação de períodos ou áreas onde esse tipo de viagem é mais comum e eficaz.

4. **Detecção de Anomalias em Tarifas (Anomaly Detection)**
   - Objetivo: Identificar viagens com tarifas anômalas ou discrepantes.
   - Métodos: Isolation Forest, One-Class SVM.
   - Resultado Esperado: Detecção de viagens com tarifas fora do padrão, ajudando a identificar possíveis erros ou fraudes.

5. **Análise Geoespacial de Áreas de Pickup e Dropoff (Visualização de Dados)**
   - Objetivo: Visualizar e analisar os padrões geoespaciais de origem e destino das viagens.
   - Métodos: Ferramentas de visualização como Folium, GeoPandas, Mapbox.
   - Resultado Esperado: Mapas interativos que mostram os pontos de acesso de origem e destino, ajudando a entender melhor a distribuição geográfica das viagens.

6. **Modelagem de Receita por Viagem (Regressão)**
   - Objetivo: Prever o total da receita gerada por viagem com base em várias características.
   - Métodos: Regressão Linear, Random Forest Regressor, Gradient Boosting Regressor.
   - Resultado Esperado: Um modelo que pode prever o total da receita esperado para uma viagem, considerando fatores como a distância da viagem, as áreas de origem e destino, e se a viagem foi compartilhada.

7. **Análise de Eficácia das Cobranças Adicionais (Classificação)**
   - Objetivo: Determinar o impacto das cobranças adicionais no total da viagem.
   - Métodos: Algoritmos de classificação para analisar a relação entre cobranças adicionais e total da viagem.
   - Resultado Esperado: Entendimento de como as cobranças adicionais influenciam o valor total da viagem, ajudando a otimizar estratégias de cobrança.

8. **Otimização de Rotas (Roteamento)**
   - Objetivo: Encontrar as rotas mais eficientes para economizar tempo e combustível.
   - Métodos: Algoritmos de roteamento como Dijkstra ou A*, Redes Neurais.
   - Resultado Esperado: Rotas otimizadas que minimizam o tempo de viagem e o consumo de combustível.

9. **Gerenciamento de Frota (Previsão)**
   - Objetivo: Utilizar a frota de veículos de maneira eficiente, ajustando a disponibilidade conforme a demanda.
   - Métodos: Modelos de previsão como ARIMA ou LSTM, Algoritmos de alocação dinâmica.
   - Resultado Esperado: Melhor utilização da frota, com ajuste dinâmico da disponibilidade de veículos conforme a demanda.

10. **Análise de Demanda (Séries Temporais e Redes Neurais)**
    - Objetivo: Identificar e prever picos de demanda para ajustar a operação.
    - Métodos: Modelos de séries temporais como ARIMA, Prophet, Redes Neurais Recurrentes (RNN) como LSTM.
    - Resultado Esperado: Previsão precisa dos picos de demanda, permitindo ajustes operacionais e alocação de recursos.

11. **Eficiência Operacional (Regressão)**
    - Objetivo: Avaliar e melhorar a eficiência operacional das viagens.
    - Métodos: Análise de Regressão, Algoritmos de otimização.
    - Resultado Esperado: Identificação de fatores que impactam a eficiência e implementação de melhorias operacionais.

12. **Planejamento Urbano (Clusterização e Visualização Geoespacial)**
    - Objetivo: Identificar pontos de acesso de origem e destino para melhorar o planejamento urbano.
    - Métodos: Clusterização de pontos de acesso usando K-means ou DBSCAN, Visualização geoespacial com Folium ou GeoPandas.
    - Resultado Esperado: Identificação de áreas de alta atividade para melhor alocação de recursos e infraestrutura.

13. **Análise de Custos (Regressão)**
    - Objetivo: Comparar custos operacionais com receitas para identificar oportunidades de redução de custos.
    - Métodos: Modelos de Regressão, Algoritmos de detecção de anomalias.
    - Resultado Esperado: Entendimento claro dos custos operacionais e identificação de áreas para redução de custos e aumento da rentabilidade.

## Dados Utilizados

O dataset utilizado contém as seguintes colunas:
- ID da Viagem
- Horário de Início da Viagem
- Horário de Fim da Viagem
- Segundos da Viagem
- Milhas da Viagem
- Área Censitária de Origem
- Área Censitária de Destino
- Área Comunitária de Origem
- Área Comunitária de Destino
- Tarifa
- Gorjeta
- Cobranças Adicionais
- Total da Viagem
- Viagem Compartilhada Autorizada
- Viagens Agrupadas
- Latitude do Centroide de Origem
- Longitude do Centroide de Origem
- Localização do Centroide de Origem
- Latitude do Centroide de Destino
- Longitude do Centroide de Destino
- Localização do Centroide de Destino

## Tecnologias Utilizadas

- **Python**: Linguagem de programação principal.
- **Pandas**: Para manipulação e análise de dados.
- **Scikit-learn**: Para implementação de algoritmos de Machine Learning.
- **Folium**: Para visualizações geoespaciais.
- **GeoPandas**: Para manipulação de dados geoespaciais.
- **Matplotlib/Seaborn**: Para visualizações de dados.

## Como Executar o Projeto

1. Clone o repositório:
   ```bash
   git clone https://github.com/seu-usuario/projeto-logistica-transportadora.git

2. Instale as dependências:
pip install -r requirements.txt

3. Execute os notebooks de análise:
jupyter notebook

## Contribuições
Contribuições são bem-vindas! Por favor, abra uma issue ou envie um pull request com melhorias e sugestões.

## Licença
Este projeto está licenciado sob a Licença MIT. Veja o arquivo LICENSE para mais detalhes.
