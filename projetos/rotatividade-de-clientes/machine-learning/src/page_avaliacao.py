import streamlit as st
import subprocess
import os
import sys

# Caminho do diretório onde o script de avaliação está localizado
script_dir = "D:/Github/data-science/projetos/rotatividade-de-clientes/machine-learning/src/"
script_path = os.path.join(script_dir, "evaluation.py")

# Função principal para exibir a página de avaliação
def main():
    # Centralizar o título "Avaliação do Modelo de Rotatividade de Clientes" com fundo destacado
    st.markdown("""
    <style>
        .title {
            background-color: #ff4b4b;
            color: white;
            padding: 10px;
            border-radius: 5px;
            text-align: center;
            font-size: 32px;
            margin-top: 20px;
        }
        .botao {
            margin-top: 30px;
        }
    </style>
    <div class="title">Avaliação do Modelo de Rotatividade de Clientes</div>
    """, unsafe_allow_html=True)

    # Função para rodar o script de avaliação
    def executar_avaliacao():
        """
        Executa o script evaluation.py e exibe os resultados.
        """
        # Verifica se o script existe antes de tentar executá-lo
        if not os.path.exists(script_path):
            st.error(f"Script de avaliação não encontrado em {script_path}. Verifique o caminho.")
            return

        try:
            # Executa o script e captura a saída
            resultado = subprocess.run(
                [sys.executable, script_path],  # Usar o Python do ambiente atual
                capture_output=True,
                text=True,
                check=True
            )
            
            # Exibe stdout e stderr do script
            if resultado.stdout:
                st.subheader("Relatório de Avaliação (stdout):")
                st.code(resultado.stdout)
            if resultado.stderr:
                st.subheader("Erros e Logs (stderr):")
                st.code(resultado.stderr)
            
            if not resultado.stdout and not resultado.stderr:
                st.warning("A avaliação foi concluída, mas não houve saída para exibir.")

        except subprocess.CalledProcessError as e:
            # Exibe qualquer erro ocorrido durante a execução
            st.error("Erro ao executar a avaliação.")
            st.code(e.stderr)

    # Botão para iniciar a avaliação com a classe "botao" para espaçamento
    st.markdown('<div class="botao"></div>', unsafe_allow_html=True)
    if st.button("Executar Avaliação"):
        executar_avaliacao()

if __name__ == "__main__":
    main()
