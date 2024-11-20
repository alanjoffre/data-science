import joblib
import yaml
import logging
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Carregar arquivo de configuração
config_path = 'D:\\Github\\data-science\\projetos\\analise-de-sentimentos\\nlp\\config\\config.yaml'
with open(config_path, 'r') as file:
    config = yaml.safe_load(file)

# Configuração de Logs
logging.basicConfig(level=logging.INFO)

def test_modelar_dados():
    model_path = config['directories']['models'] + 'modelo_sentimentos.joblib'
    try:
        model = joblib.load(model_path)
        assert model is not None, "Modelo não foi encontrado"
        assert isinstance(model.named_steps['tfidf'], TfidfVectorizer), "O vetor TF-IDF não está correto"
        assert isinstance(model.named_steps['clf'], LogisticRegression), "O classificador não está correto"
        logging.info("Teste de modelagem de dados: SUCESSO!")
    except AssertionError as e:
        logging.error(f"Teste de modelagem de dados: FALHA! Erro: {e}")
    except Exception as e:
        logging.error("Teste de modelagem de dados: FALHA! Erro: %s", e, exc_info=True)

if __name__ == "__main__":
    test_modelar_dados()
