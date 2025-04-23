# Projeto de Reconhecimento Óptico de Caracteres (OCR) com Tesseract

Este projeto visa realizar o reconhecimento óptico de caracteres (OCR) em imagens usando a ferramenta Tesseract. O OCR é realizado em imagens de texto e os resultados são extraídos e salvos em arquivos de texto e banco de dados.

## Funcionalidades

- Processamento de múltiplas imagens com OCR.
- Ajuste automático do modo de segmentação de página (PSM) baseado no tamanho da imagem.
- Pré-processamento de imagem utilizando denoising para melhorar a qualidade do OCR.
- Execução do OCR em paralelo para maior performance.
- Armazenamento dos resultados em banco de dados SQLite.
- Geração de relatórios detalhados sobre o processamento.

## Estrutura de Diretórios

reconhecimento_optico_de_caracteres_tesseract/ ├── config/ │ └── config.yaml # Arquivo de configuração ├── data/ │ ├── processed/ # Resultados do OCR em arquivos .txt │ └── raw/ # Imagens originais para o processamento ├── machine-learning/ │ ├── src/ # Código-fonte do OCR │ ├── reports/ # Relatórios gerados e imagens ├── README.md # Este arquivo └── requirements.txt # Dependências do projeto


## Dependências

Este projeto utiliza as seguintes bibliotecas Python:

- `opencv-python` - Para o processamento de imagem.
- `pytesseract` - Para o reconhecimento de texto usando Tesseract OCR.
- `pyyaml` - Para carregar configurações de arquivos YAML.
- `concurrent.futures` - Para processamento paralelo de imagens.
- `sqlite3` - Para armazenamento dos resultados em banco de dados.

Você pode instalar as dependências utilizando o `pip`:
pip install -r requirements.txt

## Arquivo requirements.txt:
opencv-python
pytesseract
pyyaml

## Configuração
Antes de executar o projeto, é necessário configurar os caminhos dos diretórios e o caminho do executável do Tesseract no arquivo config.yaml.

Exemplo de configuração:

input_dir: "D:/Github/data-science/projetos/reconhecimento_optico_de_caracteres_tesseract/machine-learning/reports/figures"
output_dir: "D:/Github/data-science/projetos/reconhecimento_optico_de_caracteres_tesseract/machine-learning/data/processed"
tesseract_cmd: "C:/Program Files/Tesseract-OCR/tesseract.exe"

## Parâmetros:
- input_dir: Diretório onde as imagens a serem processadas estão localizadas.
- output_dir: Diretório onde os resultados do OCR serão salvos.
- tesseract_cmd: Caminho para o executável do Tesseract em sua máquina.

## Como Executar

- 1 - Configuração do Tesseract: Certifique-se de ter o Tesseract OCR instalado no seu sistema. Caso não tenha, baixe e instale o Tesseract a partir do site oficial.

- 2 - Alteração do caminho do Tesseract: No arquivo config.yaml, defina o caminho para o executável do Tesseract conforme a sua instalação local.

- 3 - Alteração do diretório de imagens e saída: Defina os diretórios de entrada e saída no arquivo config.yaml.

## Para rodar o OCR:
Execute o script Python principal:
python machine-learning/src/ocr.py

O script processará as imagens no diretório de entrada, executará o OCR e salvará os resultados em arquivos .txt no diretório de saída, além de armazená-los em um banco de dados SQLite. O processamento será feito em paralelo para melhorar a performance.

## Relatórios
Após o processamento, um relatório será gerado automaticamente, contendo informações sobre o número de imagens processadas, falhas e sucesso no processo de OCR. O relatório será salvo no diretório de saída.

## Como Funciona

- 1 - Carregamento de Imagens: O código carrega as imagens do diretório de entrada.

- 2 - Pré-processamento da Imagem: As imagens são convertidas para escala de cinza e um filtro bilateral é aplicado para melhorar a qualidade da imagem antes de realizar o OCR.

- 3 - OCR: O OCR é realizado usando o Tesseract. O modo de segmentação de página (PSM) é ajustado dinamicamente com base no tamanho da imagem.

- 4 - Armazenamento dos Resultados: O texto extraído é armazenado em arquivos .txt e também em um banco de dados SQLite para futuras consultas.

- 5 - Relatório: Ao final, um relatório sobre o status do processamento é gerado.

## Banco de Dados
Os resultados do OCR são armazenados em um banco de dados SQLite, o qual pode ser consultado para verificar os resultados de OCR anteriores.

## Contribuições
Contribuições são bem-vindas! Se você quiser melhorar este projeto, envie um pull request. Por favor, siga o padrão de código e adicione testes para novas funcionalidades.

# Licença
Este projeto está licenciado sob a MIT License - veja o arquivo LICENSE para mais detalhes.