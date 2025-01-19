# Projeto de Reconhecimento Facial e Detecção de Objetos

Este projeto é uma solução avançada que integra reconhecimento facial, detecção de objetos e funcionalidades de monitoramento em tempo real. Ele utiliza tecnologias de aprendizado de máquina, visão computacional e desenvolvimento web para criar um sistema robusto e escalável.

## Funcionalidades Principais

1. **Reconhecimento Facial**
   - Identificação de pessoas em imagens, vídeos ou em tempo real.
   - Armazenamento de codificações faciais em um banco de dados relacional (PostgreSQL).

2. **Detecção de Objetos**
   - Uso do modelo YOLO (You Only Look Once) para identificar e classificar objetos em imagens e vídeos.

3. **API RESTful**
   - Criação de uma API com FastAPI para expor funcionalidades do sistema para integração com outros serviços.

4. **Gerador de Relatórios**
   - Relatórios detalhados em PDF com resultados de reconhecimento facial e detecção de objetos.

5. **Processamento em Tempo Real**
   - Suporte para processamento de vídeos ou captura ao vivo por webcam.

## Tecnologias Utilizadas

### **Linguagem de Programação**
- [Python 3.9+](https://www.python.org/)

### **Bibliotecas Principais**
- **[OpenCV](https://opencv.org/)**: Processamento de imagens e vídeos.
- **[face_recognition](https://pypi.org/project/face-recognition/)**: Reconhecimento facial.
- **[FER](https://github.com/justinshenk/fer)**: Reconhecimento de emoções.
- **[YOLO (Ultralytics)](https://docs.ultralytics.com/)**: Detecção de objetos.
- **[FastAPI](https://fastapi.tiangolo.com/)**: API RESTful.
- **[ReportLab](https://www.reportlab.com/opensource/)**: Geração de relatórios em PDF.
- **[SQLite/PostgreSQL](https://www.postgresql.org/)**: Armazenamento de codificações faciais.

### **Programas Adicionais**
- **PostgreSQL**: Banco de dados relacional.
- **MongoDB (opcional)**: Banco de dados NoSQL.

## Requisitos do Sistema

### **Softwares Necessários**
- Python 3.9+
- PostgreSQL
- MongoDB (opcional)

### **Instalação de Bibliotecas**
Execute o comando abaixo para instalar todas as dependências:
```bash
pip install -r requirements.txt
```

## Estrutura do Projeto
```plaintext
projeto-reconhecimento-facial/
│
├── main.py                # Arquivo principal para execução do projeto
├── db_manager.py          # Gerenciamento do banco de dados
├── object_detection.py    # Módulo de detecção de objetos
├── face_recognition.py    # Módulo de reconhecimento facial
├── api.py                 # API RESTful com FastAPI
├── reports.py             # Geração de relatórios em PDF
├── test_images/           # Diretório de imagens de teste
├── requirements.txt       # Dependências do projeto
└── README.md              # Documentação do projeto
```

## Configuração e Execução

### **Passo 1: Configuração do Banco de Dados**
1. Instale o PostgreSQL e crie uma base de dados chamada `face_recognition`.
2. Atualize as credenciais no arquivo `db_manager.py`.

### **Passo 2: Preparar o Ambiente**
1. Instale todas as dependências usando o `requirements.txt`.
2. Certifique-se de que os arquivos de imagem estão no diretório correto.

### **Passo 3: Executar o Sistema**
- Para reconhecimento facial:
  ```bash
  python main.py
  ```
- Para iniciar a API:
  ```bash
  uvicorn api:app --reload
  ```

## Uso da API

- **Endpoint Principal**: `/recognize`
  - Método: `POST`
  - Corpo da Requisição:
    ```json
    {
        "image_path": "caminho/para/imagem.jpg"
    }
    ```
  - Resposta:
    ```json
    {
        "recognized_faces": ["Nome da Pessoa"],
        "objects_detected": ["Objeto 1", "Objeto 2"]
    }
    ```

## Contribuição
Contribuições são bem-vindas! Siga os passos abaixo para colaborar:
1. Faça um fork do repositório.
2. Crie uma nova branch: `git checkout -b minha-branch`.
3. Envie suas alterações: `git commit -am 'Minha contribuição'`.
4. Faça um push: `git push origin minha-branch`.
5. Abra um Pull Request.

## Licença
Este projeto está licenciado sob a licença MIT. Veja o arquivo [LICENSE](LICENSE) para mais detalhes.
