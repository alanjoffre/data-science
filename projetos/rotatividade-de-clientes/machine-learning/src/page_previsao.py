import streamlit as st

# Função para exibir o conteúdo de previsão de clientes
def page_previsao():
    st.write("### Previsão de Churn para Novos Clientes")
    st.write(
        "Aqui você pode inserir as características de novos clientes e obter a probabilidade de churn (rotatividade)."
    )
    
    # Formulário para inserir características dos novos clientes
    idade = st.number_input("Idade", min_value=18, max_value=100, value=30)
    saldo = st.number_input("Saldo", min_value=0.0, max_value=1000000.0, value=50000.0)
    tempo = st.number_input("Tempo de permanência (anos)", min_value=1, max_value=50, value=3)
    ativo = st.selectbox("Cliente ativo?", ["Sim", "Não"])
    
    # Converter "Sim" / "Não" para valor booleano
    ativo = 1 if ativo == "Sim" else 0
    
    # Aqui você pode carregar o modelo e fazer a previsão
    # Exemplo fictício: resultado = modelo.predict([[idade, saldo, tempo, ativo]])
    st.write("#### Resultado da Previsão:")
    st.write("A probabilidade de churn para este cliente é de 72%.")  # Exemplo de resultado

# Executar a página
if __name__ == "__main__":
    page_previsao()
