# Projeto: Analise de Sentimentos X

# Etapa: Carregamento / Pre-processamento dos Dados
- Instalar as bibliotecas necessárias
- Importar bibliotecas
- Carregar as configurações do arquivo config.yaml
- Configurar logging
- Função para configurar o logger
- Baixar recursos necessários do NLTK: stopwords, punkt, wordnet
- Carregar o dataset do arquivo Parquet usando Dask
- Renomear as colunas
- Verificação inicial das colunas e dados
- Função para tratar a coluna created_at com fusos horários desconhecidos
- Função para analisar sentimentos usando VADER
- Função para limpar texto conforme configurações do config.yaml
- Função para remover stopwords
- Função para tokenizar texto
- Função para lematizar texto
- Função para remover emojis usando emoji.demojize
- Função para processar texto usando Dask Delayed
- Aplicar as funções de limpeza no dataset
- Tratar valores ausentes e duplicatas
- Visualizar as mudanças
- Converter o Dask DataFrame para Pandas DataFrame para análise exploratória
- Salvar o dataset limpo
- Carregar o dataset do arquivo Parquet usando Dask
- Identificar colunas com NaNs
- Mostrar a contagem de NaNs por coluna
- Tratar valores NaN nas colunas numéricas
- Verificar se ainda há NaNs
- Salvar o dataset limpo novamente

# Etapa: Analise Exploratória de Dados (EDA) 
- Instalar as bibliotecas necessárias
- Importar as bibliotecas
- Carregar as configurações do arquivo config.yaml
- Configurar logging
- Função para configurar o logger
- Carregar o dataset do arquivo Parquet usando Dask - Dataset, limpo e final da etapa anterior
- Verificação inicial das colunas e dados
- Converter a coluna 'created_at' para datetime
- Contar valores NaN na coluna 'created_at' antes do tratamento
- Converter 'created_at' para timestamp numérico (segundos desde a época)
- Calcular a média da coluna 'created_at_numeric'
- Substituir NaN na coluna 'created_at_numeric' pela média calculada
- Converter 'created_at_numeric' de volta para datetime
- Contar valores NaN na coluna 'created_at' após o tratamento
- Verificar as primeiras linhas para garantir que os NaN foram tratados
- Calcular a média e desvio padrão dos timestamps numéricos
- Remover a coluna 'created_at_numeric' após cálculo do desvio padrão
- Validação final para assegurar que não existem NaN na coluna 'created_at'
- Análise exploratória manual
- Converter a coluna 'created_at' para string para evitar erros no salvamento
- Salvar as saídas em arquivos Parquet e CSV
- Visualização dos dados
- Log_info('Visualização da distribuição de datas salva.')
- Log_info('Visualização da contagem de sentimentos salva.')
- Log_info('Início da extração e análise de hashtags e menções.')
- Função para identificar as palavras mais frequentes
- Função para extrair hashtags
- Função para extrair menções
- Aplicar extração de hashtags e menções
- Converter para Pandas DataFrame
- Contar hashtags mais frequentes
- Contar menções mais frequentes
- Visualizações de Hashtags e Menções
- Log_info('Visualização das Top 20 Menções salva com sucesso.')
- Função para identificar os bigrams mais comuns
- Função para identificar os trigrams mais comuns
- Aplicar análise de bigrams e trigrams
- Log_info('Bigrams mais comuns salvos com sucesso.')
- Log_info('Trigrams mais comuns salvos com sucesso.')
- Função para analisar sentimentos usando VADER
- Aplicar análise de sentimentos
- Converter para Pandas DataFrame para salvar resultados
- Salvar resultados de análise de sentimentos em arquivos Parquet e CSV
- Visualização da Análise de Sentimentos
- Garantir que todos os dados estejam limpos e salvos corretamente
- Recarregar o DataFrame limpo para verificação final
- Verificar as primeiras linhas e as colunas do DataFrame
- Verificar se não há valores NaN no DataFrame final
- Salvar o DataFrame final em novos arquivos Parquet e CSV para garantir que as mudanças foram aplicadas

### Projeto em andamento: 15.11.2024

# Licença
- Este projeto está licenciado sob a Licença MIT - veja o arquivo LICENSE para mais detalhes.