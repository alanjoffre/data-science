# Previsão de Mercado de Ações

- Este projeto utiliza Spark e redes neurais LSTM (Long Short-Term Memory) para prever o comportamento do mercado de ações. O objetivo é fornecer previsões precisas dos preços futuros das ações com base em dados históricos, auxiliando investidores e analistas financeiros a tomarem decisões mais informadas. O uso de Spark permite o processamento eficiente de grandes volumes de dados, enquanto as LSTMs são ideais para capturar padrões temporais e dependências de longo prazo nas séries temporais de preços das ações.

- O projeto abrange todas as etapas do pipeline de dados, incluindo a coleta e preparação dos dados, construção do modelo, treinamento, avaliação e previsão. Também inclui a criação de novas features, como médias móveis e índices de força relativa (RSI), para melhorar a precisão das previsões. Com este sistema, esperamos aumentar a precisão nas previsões do mercado de ações e proporcionar uma ferramenta valiosa para estratégias de investimento.

# Objetivos do Projeto:
- Coletar Dados Históricos: Obter dados históricos de preços de ações de fontes confiáveis.
- Preparar e Limpar os Dados: Tratar valores ausentes, remover inconsistências e normalizar os dados.
- Criar Features Adicionais: Incluir métricas como médias móveis e RSI para enriquecer o conjunto de dados.
- Construir e Treinar o Modelo LSTM: Utilizar PySpark para processar dados e Keras/TensorFlow para construir o modelo de LSTM.
- Avaliar o Desempenho do Modelo: Utilizar métricas como RMSE, MAE e R² para avaliar a precisão das previsões.
- Implementar e Fazer Previsões: Usar o modelo treinado para prever preços futuros das ações e comparar com os dados reais.

# Benefícios Esperados:
- Precisão Aprimorada: Utilização de técnicas avançadas de redes neurais para capturar padrões complexos e temporais.
- Processamento Eficiente: Utilização do Spark para lidar com grandes volumes de dados de forma eficiente.
- Decisões Informadas: Ferramenta valiosa para analistas e investidores tomarem decisões mais fundamentadas no mercado de ações.