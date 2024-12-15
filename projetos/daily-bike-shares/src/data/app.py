import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt

# Função para pré-processar dados de entrada para previsão
def preprocess_input(input_data, model_feature_names):
    input_data['temp_hum'] = input_data['temp'] * input_data['hum']
    input_data['temp_squared'] = input_data['temp'] ** 2
    input_data['weekday_weekend'] = input_data['workingday'].apply(lambda x: 1 if x == 0 else 0)

    # Codificação one-hot para as variáveis categóricas
    input_data = pd.get_dummies(input_data, columns=['season', 'weathersit', 'mnth', 'weekday'], drop_first=True)

    # Garantir que todas as colunas que o modelo espera estão presentes, mesmo que ausentes
    input_data = input_data.reindex(columns=model_feature_names, fill_value=0)

    return input_data

# Função para fazer previsões
def make_prediction(input_data):
    model_path = r"D:\Github\data-science\projetos\daily-bike-shares\src\models\xgb_model.joblib"
    model = joblib.load(model_path)

    # Preprocessar os dados de entrada
    input_df = preprocess_input(input_data, model.get_booster().feature_names)

    # Prever
    prediction = model.predict(input_df)
    return prediction

# Função para exibir gráficos (gráfico de linhas)
def plot_prediction_distribution(prediction):
    plt.figure(figsize=(10, 5))
    plt.plot(['Previsão'], [prediction[0]], marker='o', color='skyblue', label='Previsão de Aluguel')
    plt.ylabel('Número de Aluguéis')
    plt.title('Distribuição da Previsão de Aluguel de Bicicletas')
    plt.ylim(0, max(10, prediction[0] + 5))  # Ajusta o limite superior do gráfico
    plt.grid()
    plt.legend()
    st.pyplot(plt)  # Mostrar o gráfico no Streamlit
    plt.close()  # Fechar a figura para evitar duplicação de gráficos

# Aplicativo Streamlit
def main():
    st.set_page_config(page_title="Previsão de Aluguel de Bicicletas", layout="wide")
    
    st.title("Previsão de Aluguel de Bicicletas")
    st.markdown("""
    Preencha os parâmetros abaixo para prever a quantidade de aluguéis de bicicletas para um dia específico. 
    Utilize os sliders e selecione as opções de acordo com a sua necessidade.
    """)

    # Sidebar para entradas do usuário
    st.sidebar.header("Parâmetros de Entrada")
    year = st.sidebar.selectbox("Selecione o ano:", options=[0, 1], format_func=lambda x: "2011" if x == 0 else "2012")
    holiday = st.sidebar.selectbox("É feriado?", options=[0, 1], format_func=lambda x: "Não" if x == 0 else "Sim")
    working_day = st.sidebar.selectbox("É dia útil?", options=[0, 1], format_func=lambda x: "Não" if x == 0 else "Sim")

    temp = st.sidebar.slider("Temperatura Normalizada (0.0 a 1.0)", min_value=0.0, max_value=1.0, value=0.5)
    atemp = st.sidebar.slider("Temperatura Ajustada Normalizada (0.0 a 1.0)", min_value=0.0, max_value=1.0, value=0.5)
    hum = st.sidebar.slider("Umidade Normalizada (0.0 a 1.0)", min_value=0.0, max_value=1.0, value=0.4)
    windspeed = st.sidebar.slider("Velocidade do Vento Normalizada (0.0 a 1.0)", min_value=0.0, max_value=1.0, value=0.1)

    season = st.sidebar.selectbox("Selecione a Estação do Ano:", options=[1, 2, 3, 4], format_func=lambda x: {1: "Inverno", 2: "Primavera", 3: "Verão", 4: "Outono"}[x])
    weathersit = st.sidebar.selectbox("Selecione a Situação do Tempo:", options=[1, 2, 3, 4], format_func=lambda x: {1: "Limpo", 2: "Nublado", 3: "Chuva", 4: "Neve"}[x])
    mnth = st.sidebar.selectbox("Selecione o Mês:", options=list(range(1, 13)), format_func=lambda x: f"Mês {x}")
    weekday = st.sidebar.selectbox("Selecione o Dia da Semana:", options=list(range(0, 7)), format_func=lambda x: {0: "Domingo", 1: "Segunda", 2: "Terça", 3: "Quarta", 4: "Quinta", 5: "Sexta", 6: "Sábado"}[x])

    # Criar um botão para fazer a previsão
    if st.sidebar.button("Fazer Previsão"):
        input_data = pd.DataFrame({
            'yr': [year],
            'holiday': [holiday],
            'workingday': [working_day],
            'temp': [temp],
            'atemp': [atemp],
            'hum': [hum],
            'windspeed': [windspeed],
            'season': [season],
            'weathersit': [weathersit],
            'mnth': [mnth],
            'weekday': [weekday],
        })

        pred = make_prediction(input_data)
        st.success(f"Previsão de aluguel: {pred[0]:.2f} bicicletas")

        # Mostrar os parâmetros de entrada
        st.markdown("### Parâmetros Usados na Previsão:")
        st.write(input_data)

        # Plotar distribuição da previsão
        plot_prediction_distribution(pred)

if __name__ == "__main__":
    main()
