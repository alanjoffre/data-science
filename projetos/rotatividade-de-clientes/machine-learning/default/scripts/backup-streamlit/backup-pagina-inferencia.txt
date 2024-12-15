import streamlit as st
import subprocess
import os
import sys

# Caminho do diretório onde o script de inferência está localizado
script_dir = "D:/Github/data-science/projetos/rotatividade-de-clientes/machine-learning/src/"
script_path = os.path.join(script_dir, "inferencia.py")

# Função principal para exibir a página de inferência
def main():
    # Definindo o estilo para o título
    st.markdown(""" 
    <style>
        .pipeline-title {
            background-color: #ff4b4b;
            color: white;
            padding: 10px;
            border-radius: 5px;
            text-align: center;
            font-size: 24px;
        }
        .section-title {
            font-size: 18px;
            font-weight: bold;
            margin-top: 10px;
            margin-bottom: 10px;
        }
        .content-text {
            font-size: 14px;
        }
    </style>
    <div class="pipeline-title">Inferência do Modelo de Rotatividade de Clientes</div>
    """, unsafe_allow_html=True)

    # Função para rodar o script de inferência
    def executar_inferencia():
        """
        Executa o script inferencia.py e exibe os resultados.
        """
        # Verifica se o script existe antes de tentar executá-lo
        if not os.path.exists(script_path):
            st.error(f"Script de inferência não encontrado em {script_path}. Verifique o caminho.")
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
                st.subheader("Relatório de Inferência (stdout):")
                st.code(resultado.stdout)
            if resultado.stderr:
                st.subheader("Erros e Logs (stderr):")
                st.code(resultado.stderr)
            
            if not resultado.stdout and not resultado.stderr:
                st.warning("A inferência foi concluída, mas não houve saída para exibir.")

        except subprocess.CalledProcessError as e:
            # Exibe qualquer erro ocorrido durante a execução
            st.error("Erro ao executar a inferência.")
            st.code(e.stderr)

    # Botão para iniciar a inferência com a classe "botao" para espaçamento
    st.markdown('<div class="botao"></div>', unsafe_allow_html=True)
    if st.button("Executar Inferência"):
        executar_inferencia()

if __name__ == "__main__":
    main()
