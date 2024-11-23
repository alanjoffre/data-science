import sys
import os
import yaml

# Adicionando o diretório src ao PYTHONPATH para garantir que a importação funcione corretamente
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

def update_config_yaml(file_path):
    # Novo conteúdo a ser adicionado no arquivo config.yaml
    new_config = {
        "directories": {
            "raw_data": "D:\\Github\\data-science\\projetos\\analise-de-sentimentos\\nlp\\data\\raw\\",
            "processed_data": "D:\\Github\\data-science\\projetos\\analise-de-sentimentos\\nlp\\data\\processed\\",
            "logs": "D:\\Github\\data-science\\projetos\\analise-de-sentimentos\\nlp\\logs"
        },
        "files": {
            "raw_dataset": "asentimentos.parquet",
            "processed_dataset": "processed_dataset.parquet",
            "processed_dataset_csv": "processed_dataset.csv",
            "log_file": "etapa1_carregamento_dados.log"
        }
    }

    try:
        # Abrindo o arquivo YAML existente
        with open(file_path, 'r') as file:
            config = yaml.safe_load(file) or {}

        # Atualizando o arquivo com os novos valores
        config.update(new_config)

        # Salvando as alterações no arquivo YAML
        with open(file_path, 'w') as file:
            yaml.dump(config, file, default_flow_style=False, allow_unicode=True)

        print(f"Arquivo '{file_path}' atualizado com sucesso!")

    except FileNotFoundError:
        print(f"O arquivo '{file_path}' não foi encontrado.")
    except Exception as e:
        print(f"Ocorreu um erro ao atualizar o arquivo: {e}")

# Caminho para o arquivo config.yaml
config_file_path = 'D:\\Github\\data-science\\projetos\\analise-de-sentimentos\\nlp\\config\\config.yaml'

# Atualizando o arquivo config.yaml
update_config_yaml(config_file_path)
