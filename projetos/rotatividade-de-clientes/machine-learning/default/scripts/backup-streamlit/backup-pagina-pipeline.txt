import streamlit as st
import subprocess
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
import io

def generate_pdf(output):
    # Cria um buffer de memória para armazenar o PDF
    buffer = io.BytesIO()

    # Cria o canvas para gerar o PDF
    c = canvas.Canvas(buffer, pagesize=letter)
    
    # Define o título e formatação
    c.setFont("Helvetica-Bold", 14)  # Fonte um pouco menor para ajustar o conteúdo
    c.drawString(100, 750, "Relatório de Execução do Pipeline de Produção")

    # Define o corpo do texto
    c.setFont("Helvetica", 10)  # Fonte reduzida para ajustar as linhas

    y_position = 730  # Posição inicial do texto
    
    # Mensagens de resumo do pipeline
    etapas = {
        "Pré-processamento de Dados": "✅ Etapa de preparação de dados concluída com sucesso. Dados transformados e limpos.",
        "Treinamento e Validação do Modelo": "✅ Modelos treinados e validados com sucesso. Melhor modelo selecionado.",
        "Avaliação e Métricas": "📊 Métricas de avaliação calculadas. Resultados: acurácia e F1-Score apresentados.",
        "Implantação do Modelo": "🚀 Modelo exportado para produção. Pronto para previsões em novos dados."
    }
    
    # Escreve as etapas no PDF
    for etapa, mensagem in etapas.items():
        c.drawString(100, y_position, f"{etapa}: {mensagem}")
        y_position -= 20  # Ajusta a posição para a próxima linha

    # Adiciona uma linha de separação
    c.line(100, y_position - 10, 500, y_position - 10)

    # Detalhamento do processo
    c.drawString(100, y_position - 30, "Detalhamento do Processo:")
    y_position -= 40

    # Captura a saída do console em tempo real
    output_lines = output.split("\n")  # Pegando todo o conteúdo
    for line in output_lines:
        if y_position > 50:  # Verifica se ainda há espaço na página
            c.drawString(100, y_position, line)
            y_position -= 12  # Ajusta a altura para a próxima linha

    # Adiciona uma linha de separação final
    c.line(100, y_position - 10, 500, y_position - 10)

    # Salva o conteúdo no buffer
    c.save()

    # Retorna o conteúdo do PDF
    buffer.seek(0)
    return buffer

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
    <div class="pipeline-title">Pipeline de Produção - Análise de Churn de Clientes</div>
    """, unsafe_allow_html=True)

    # Texto de introdução ao pipeline de produção
    st.write("Esta página apresenta o pipeline de produção utilizado para a análise de churn de clientes, desde o processamento de dados até a geração de previsões.")

    # Seções explicativas do pipeline
    st.markdown('<p class="section-title">1. Pré-processamento de Dados</p>', unsafe_allow_html=True)
    st.write("Transformação, limpeza e codificação de dados para garantir que o conjunto esteja pronto para o modelo.")

    st.markdown('<p class="section-title">2. Treinamento e Validação do Modelo</p>', unsafe_allow_html=True)
    st.write("Treinamento e ajuste dos modelos usando validação cruzada e ajustes de hiperparâmetros.")

    st.markdown('<p class="section-title">3. Avaliação e Métricas</p>', unsafe_allow_html=True)
    st.write("Avaliação das métricas do modelo, incluindo acurácia e F1-Score.")

    st.markdown('<p class="section-title">4. Implantação do Modelo</p>', unsafe_allow_html=True)
    st.write("Exportação do modelo final para uso em produção.")

    # Botão para executar o pipeline
    if st.button("Executar Pipeline"):
        with st.spinner("Processando o pipeline, por favor aguarde..."):
            try:
                # Executa o arquivo pipeline.py e captura o output em tempo real
                result = subprocess.run(["python", "pipeline.py"], capture_output=True, text=True)
                output = result.stdout  # Captura o output do processo
                
                # Exibe o resumo organizado por etapas
                st.success("Pipeline executado com sucesso!")
                
                st.write("### Sumário dos Resultados da Execução do Pipeline")
                
                # Exibe as principais etapas e suas mensagens com ícones e cores
                etapas = {
                    "Pré-processamento de Dados": "✅ Etapa de preparação de dados concluída com sucesso. Dados transformados e limpos.",
                    "Treinamento e Validação do Modelo": "✅ Modelos treinados e validados com sucesso. Melhor modelo selecionado.",
                    "Avaliação e Métricas": "📊 Métricas de avaliação calculadas. Resultados: acurácia e F1-Score apresentados.",
                    "Implantação do Modelo": "🚀 Modelo exportado para produção. Pronto para previsões em novos dados."
                }
                
                # Exibe cada etapa com a mensagem correspondente
                for etapa, mensagem in etapas.items():
                    st.markdown(f"**{etapa}**")
                    st.write(mensagem)

            except Exception as e:
                st.error(f"Erro ao executar o pipeline: {e}")

if __name__ == "__main__":
    main()
