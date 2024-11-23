import os
import sys
import yaml
import tempfile

# Adicionando o diretório src ao sys.path para garantir a importação correta
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

# Importando a função corretamente após ajustar o sys.path
from etapa0_inicio_do_projeto import update_config_yaml

def test_update_config_yaml():
    # Criar um arquivo temporário para simular o config.yaml
    with tempfile.NamedTemporaryFile(delete=False, suffix='.yaml') as temp_file:
        temp_file_path = temp_file.name

    try:
        # Criar conteúdo inicial para o arquivo YAML
        initial_config = {
            "directories": {
                "raw_data": "old/path/raw",
                "processed_data": "old/path/processed"
            },
            "files": {
                "raw_dataset": "old_dataset.parquet"
            }
        }

        # Escrever o conteúdo inicial no arquivo temporário
        with open(temp_file_path, 'w') as file:
            yaml.dump(initial_config, file, default_flow_style=False, allow_unicode=True)

        # Rodar a função de atualização
        update_config_yaml(temp_file_path)

        # Ler o conteúdo atualizado
        with open(temp_file_path, 'r') as file:
            updated_config = yaml.safe_load(file)

        # Verificar se o conteúdo foi atualizado corretamente
        expected_config = {
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

        # Asserting if the updated config matches the expected one
        assert updated_config == expected_config, f"Configuração incorreta: {updated_config}"

        print("Teste concluído com sucesso! O arquivo foi atualizado corretamente.")
    finally:
        # Remover o arquivo temporário
        os.remove(temp_file_path)

if __name__ == "__main__":
    test_update_config_yaml()
