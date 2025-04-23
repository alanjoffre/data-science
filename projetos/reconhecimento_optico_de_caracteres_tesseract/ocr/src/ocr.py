# Baixar o instalador
# https://github.com/UB-Mannheim/tesseract/wiki

# Adicione o dieretorio Tesserat_OCR a variavel de ambiente PATH
# Adicione o diretório do Tesseract-OCR à variável de ambiente PATH. Para fazer isso:
# No Windows, abra as "Configurações do Sistema Avançado".
# Vá em Variáveis de Ambiente.
# Na seção Variáveis do Sistema, localize a variável Path e edite-a.
# Adicione o seguinte caminho:
# C:\Program Files\Tesseract-OCR

# Instalar as bibliotecas: opencv-python==4.6.0.66, opencv-contrib-python==4.6.0.66 e pytesseract

import cv2
import pytesseract
import os
import logging
import concurrent.futures
import sqlite3
from datetime import datetime
import yaml

# Função para carregar configurações do arquivo config.yaml
def carregar_config(config_path):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

# Função para validar se as imagens são válidas
def validar_imagens(input_dir):
    imagens = [os.path.join(input_dir, f) for f in os.listdir(input_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    if not imagens:
        logging.warning(f"Nenhuma imagem encontrada em {input_dir}.")
    return imagens

# Função para definir o PSM dinamicamente com base nas dimensões da imagem
def definir_psm(imagem):
    altura, largura = imagem.shape[:2]
    if altura < 200:
        return 7  # Modo de uma linha
    else:
        return 6  # Modo de bloco de texto

# Função para aplicar denoising (filtro bilateral)
def denoising_avancado(img):
    img_denoised = cv2.bilateralFilter(img, d=9, sigmaColor=75, sigmaSpace=75)
    return img_denoised

# Função para processar cada imagem
def process_image(img_path, lang='eng', tesseract_cmd='C:\\Program Files\\Tesseract-OCR\\tesseract.exe'):
    # Carregar a imagem
    img = cv2.imread(img_path)
    if img is None:
        logging.error(f"Erro ao carregar a imagem: {img_path}.")
        return None
    
    # Converter para escala de cinza
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Aplicar denoising
    img_gray = denoising_avancado(img_gray)
    
    # Definir PSM baseado no tamanho da imagem
    psm = definir_psm(img_gray)
    
    # Configurar OCR
    pytesseract.pytesseract.tesseract_cmd = tesseract_cmd
    config = f"--psm {psm} -l {lang}"
    
    # Executar OCR
    texto_extraido = pytesseract.image_to_string(img_gray, config=config)
    
    return texto_extraido

# Função para processar imagens em paralelo
def processar_imagens_concorrente(imagens, lang, tesseract_cmd):
    with concurrent.futures.ThreadPoolExecutor() as executor:
        resultados = executor.map(lambda img_path: process_image(img_path, lang, tesseract_cmd), imagens)
    return list(resultados)

# Função para salvar os resultados no banco de dados
def salvar_no_banco(texto, imagem_path, db_path='ocr_results.db'):
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    c.execute("CREATE TABLE IF NOT EXISTS resultados (imagem_path TEXT, texto_extraido TEXT, data_processamento TEXT)")
    c.execute("INSERT INTO resultados (imagem_path, texto_extraido, data_processamento) VALUES (?, ?, ?)", 
              (imagem_path, texto, datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
    conn.commit()
    conn.close()

# Função para gerar o relatório de processamento
def gerar_relatorio(diretorio_saida, imagens_processadas, falhas):
    relatorio_path = os.path.join(diretorio_saida, "relatorio_ocr.txt")
    with open(relatorio_path, "w") as f:
        f.write(f"Relatório de OCR - Processamento Concluído\n")
        f.write(f"Data: {datetime.now()}\n")
        f.write(f"Imagens Processadas: {imagens_processadas}\n")
        f.write(f"Falhas: {falhas}\n")
    logging.info(f"Relatório gerado em: {relatorio_path}")

# Função para configurar logging
def setup_logging(log_dir):
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)  # Cria o diretório caso não exista
    
    log_file = os.path.join(log_dir, 'ocr.log')
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    logging.info("Logging configurado com sucesso.")

# Função principal de execução
def main():
    # Caminho do arquivo de configuração
    config_path = r"D:/Github/data-science/projetos/reconhecimento_optico_de_caracteres_tesseract/ocr/config/config.yaml"
    config = carregar_config(config_path)
    
    # Diretórios
    input_dir = config['input_dir']
    output_dir = config['output_dir']
    tesseract_cmd = config['tesseract_cmd']
    log_dir = config['log_dir']
    
    # Configurar logging
    setup_logging(log_dir)
    
    # Validar imagens no diretório
    imagens = validar_imagens(input_dir)
    
    if not imagens:
        logging.error("Nenhuma imagem válida encontrada. Processo abortado.")
        return
    
    # Processar as imagens em paralelo
    logging.info("Iniciando o processamento das imagens.")
    resultados = processar_imagens_concorrente(imagens, 'eng', tesseract_cmd)
    
    # Salvar os resultados no banco de dados e gerar relatório
    falhas = 0
    for img_path, texto in zip(imagens, resultados):
        if texto:
            salvar_no_banco(texto, img_path)
            with open(os.path.join(output_dir, os.path.basename(img_path) + '.txt'), 'w') as file:
                file.write(texto)
        else:
            falhas += 1
    
    # Gerar relatório final
    gerar_relatorio(output_dir, len(imagens), falhas)
    logging.info("Processamento concluído.")

# Executar o código
if __name__ == '__main__':
    main()
