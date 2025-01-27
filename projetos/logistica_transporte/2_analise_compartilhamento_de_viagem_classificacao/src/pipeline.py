import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
import joblib
from imblearn.over_sampling import SMOTE
import seaborn as sns
import matplotlib.pyplot as plt

# Objetivo do Projeto
objetivo = "Analisar o compartilhamento de viagem utilizando métodos de classificação para determinar os fatores que influenciam a probabilidade de uma viagem ser compartilhada."

# Identificação das Partes Interessadas
stakeholders = ["Gestores de Operações", "Equipes de Marketing", "Desenvolvedores de Produto"]

# Requisitos de Negócio e Técnicos
requisitos_negocio = ["Prever demanda por viagens compartilhadas", "Otimização de rotas", "Personalização da experiência do usuário"]
requisitos_tecnicos = ["Indicadores-chave de desempenho (KPIs)", "Conformidade com regulamentos de dados"]

# Carregar dados de diferentes fontes
data_raw_path = 'D:/Github/data-science/projetos/logistica_transporte/2_analise_compartilhamento_de_viagem_classificacao/data/raw/dataset_preprocessado.parquet'
data = pd.read_parquet(data_raw_path)

# Quantidade de linhas no dataset
num_linhas = data.shape[0]
print(f"Número de linhas no dataset: {num_linhas}")

# Tipos de cada coluna
tipos_colunas = data.dtypes
print("Tipos de cada coluna:\n", tipos_colunas)

# Verificação de valores ausentes
valores_ausentes = data.isnull().sum()
print("Valores ausentes por coluna:\n", valores_ausentes)

# Trabalhar com uma amostra do dataset (exemplo: 5% do dataset)
amostra = data.sample(frac=0.05, random_state=42)
amostra_path = 'D:/Github/data-science/projetos/logistica_transporte/2_analise_compartilhamento_de_viagem_classificacao/data/raw/dataset_amostra.parquet'
amostra.to_parquet(amostra_path)
print(f"Amostra do dataset salva em: {amostra_path}")

# Tratar dados ausentes - Exemplo de preenchimento com a média (para colunas numéricas) e moda (para colunas categóricas)
amostra['segundos_da_viagem'].fillna(amostra['segundos_da_viagem'].mean(), inplace=True)
amostra['milhas_da_viagem'].fillna(amostra['milhas_da_viagem'].mean(), inplace=True)
amostra['trato_do_censo_do_embarque'].fillna(amostra['trato_do_censo_do_embarque'].median(), inplace=True)
amostra['trato_do_censo_do_desembarque'].fillna(amostra['trato_do_censo_do_desembarque'].median(), inplace=True)
amostra['area_comunitaria_do_embarque'].fillna(amostra['area_comunitaria_do_embarque'].median(), inplace=True)
amostra['area_comunitaria_do_desembarque'].fillna(amostra['area_comunitaria_do_desembarque'].median(), inplace=True)
amostra['tarifa'].fillna(amostra['tarifa'].mean(), inplace=True)
amostra['gorjeta'].fillna(amostra['gorjeta'].mean(), inplace=True)
amostra['cobrancas_adicionais'].fillna(amostra['cobrancas_adicionais'].mean(), inplace=True)
amostra['total_da_viagem'].fillna(amostra['total_da_viagem'].mean(), inplace=True)
amostra['latitude_do_centroide_do_embarque'].fillna(amostra['latitude_do_centroide_do_embarque'].median(), inplace=True)
amostra['longitude_do_centroide_do_embarque'].fillna(amostra['longitude_do_centroide_do_embarque'].median(), inplace=True)
amostra['local_do_centroide_do_embarque'].fillna(amostra['local_do_centroide_do_embarque'].mode()[0], inplace=True)
amostra['latitude_do_centroide_do_desembarque'].fillna(amostra['latitude_do_centroide_do_desembarque'].median(), inplace=True)
amostra['longitude_do_centroide_do_desembarque'].fillna(amostra['longitude_do_centroide_do_desembarque'].median(), inplace=True)
amostra['local_do_centroide_do_desembarque'].fillna(amostra['local_do_centroide_do_desembarque'].mode()[0], inplace=True)

# Verificar se ainda existem valores ausentes
valores_ausentes_pos_tratamento = amostra.isnull().sum()
print("Valores ausentes após tratamento:\n", valores_ausentes_pos_tratamento)

# Eliminar outliers
amostra = amostra[amostra['milhas_da_viagem'] < 100]

# Validação de Dados
amostra['data_inicio'] = pd.to_datetime(amostra['data_inicio'], errors='coerce')
amostra['data_final'] = pd.to_datetime(amostra['data_final'], errors='coerce')
amostra = amostra.dropna(subset=['data_inicio', 'data_final'])

# Normalização e Codificação
numerical_features = ['segundos_da_viagem', 'milhas_da_viagem', 'tarifa', 'gorjeta', 'cobrancas_adicionais', 'total_da_viagem']
categorical_features = ['trato_do_censo_do_embarque', 'trato_do_censo_do_desembarque', 'area_comunitaria_do_embarque', 'area_comunitaria_do_desembarque', 'hora_inicio', 'data_inicio']

# Criação de novas features
amostra['hora_inicio'] = amostra['hora_inicio'].apply(lambda x: int(x.split(':')[0]))  # Extrair hora
amostra['dia_da_semana'] = amostra['data_inicio'].dt.weekday  # Extrair dia da semana

# Atualizar lista de features categóricas
categorical_features.append('dia_da_semana')

# Pipeline para transformação
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(), categorical_features)
    ]
)

# Aplicar pré-processamento
data_preprocessed = preprocessor.fit_transform(amostra)

# Salvar o preprocessor
preprocessor_path = 'D:/Github/data-science/projetos/logistica_transporte/2_analise_compartilhamento_de_viagem_classificacao/preprocessors/preprocessor.joblib'
joblib.dump(preprocessor, preprocessor_path)

# Verificar a integridade dos dados integrados
num_linhas_colunas = amostra.shape
print("Número de linhas e colunas no dataset:", num_linhas_colunas)

# Verificar se todas as fontes de dados foram corretamente integradas
colunas_disponiveis = amostra.columns
print("Colunas disponíveis no dataset:", colunas_disponiveis)

# Balancear a variável de destino
X = data_preprocessed
y = amostra['viagem_compartilhada_autorizada']

smote = SMOTE(random_state=42)
X_res, y_res = smote.fit_resample(X, y)

distribuicao_balanceamento = y_res.value_counts()
print("Distribuição após balanceamento:", distribuicao_balanceamento)

# Calcular a correlação entre as variáveis
correlation_matrix = amostra.corr()

# Plotar a matriz de correlação e salvar o gráfico
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
correlation_plot_path = 'D:/Github/data-science/projetos/logistica_transporte/2_analise_compartilhamento_de_viagem_classificacao/reports/figures/correlation_matrix.png'
plt.savefig(correlation_plot_path)
plt.close()
print(f"Gráfico de correlação salvo em: {correlation_plot_path}")

# Gerar um relatório descritivo
report = amostra.describe()

# Salvar o relatório em um arquivo CSV
report_path = 'D:/Github/data-science/projetos/logistica_transporte/2_analise_compartilhamento_de_viagem_classificacao/reports/relatorio_descritivo.csv'
report.to_csv(report_path)

print(f"Relatório descritivo salvo em {report_path}")
