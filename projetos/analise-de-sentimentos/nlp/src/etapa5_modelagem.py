import dask.dataframe as dd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, roc_auc_score, roc_curve
from sklearn.model_selection import GridSearchCV, train_test_split, cross_val_score
import numpy as np
import logging
import yaml
import joblib
import matplotlib.pyplot as plt
from sklearn.preprocessing import label_binarize
from itertools import cycle

# Carregar arquivo de configuração
config_path = 'D:\\Github\\data-science\\projetos\\analise-de-sentimentos\\nlp\\config\\config.yaml'
with open(config_path, 'r') as file:
    config = yaml.safe_load(file)

# Configuração de Logs
logging.basicConfig(level=config['logs']['level'],
                    format=config['logs']['format'],
                    handlers=[logging.StreamHandler(),
                              logging.FileHandler(config['directories']['logs'] + 'modelagem.log')])
logger = logging.getLogger(__name__)

def plot_roc_curve(fpr, tpr, roc_auc, classes, fig_path):
    plt.figure()
    colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
    for i, color in zip(range(len(classes)), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=2, label=f'ROC curve (class {classes[i]}) (area = {roc_auc[i]:0.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Taxa de Falsos Positivos')
    plt.ylabel('Taxa de Verdadeiros Positivos')
    plt.title('Receiver Operating Characteristic para cada classe')
    plt.legend(loc="lower right")
    plt.savefig(fig_path)
    plt.show()

def modelar_dados():
    # Carregar o dataset classificado
    classified_data_path = config['directories']['processed_data'] + config['files']['processed_dataset']
    logger.info("Carregando dataset classificado do caminho: %s", classified_data_path)

    df = dd.read_parquet(classified_data_path).compute()
    logger.info("Dataset classificado carregado com sucesso!")

    # Dividir os dados em treinamento e teste
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
    logger.info("Conjunto de dados dividido com sucesso!")

    X_train, y_train = train_df['processed_tweet'], train_df['sentimento']
    X_test, y_test = test_df['processed_tweet'], test_df['sentimento']

    # Construção do pipeline
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer()),
        ('clf', RandomForestClassifier(random_state=42))
    ])

    # Definir a grade de parâmetros para busca
    param_grid = {
        'clf__n_estimators': [100, 200, 300, 400],
        'clf__max_depth': [None, 10, 20, 30],
        'clf__min_samples_split': [2, 5, 10, 15],
        'clf__min_samples_leaf': [1, 2, 4, 6]
    }

    # Realizar busca em grade com validação cruzada
    grid_search = GridSearchCV(pipeline, param_grid, cv=5, n_jobs=-1, scoring='f1_weighted')
    grid_search.fit(X_train, y_train)
    logger.info("Melhores parâmetros encontrados: %s", grid_search.best_params_)

    # Treinar o modelo com os melhores parâmetros
    best_model = grid_search.best_estimator_
    best_model.fit(X_train, y_train)
    logger.info("Modelo treinado com os melhores parâmetros.")

    # Avaliação do modelo
    y_pred = best_model.predict(X_test)
    report = classification_report(y_test, y_pred, output_dict=True)
    logger.info(f"Relatório de classificação (Parâmetros Otimizados):\n {classification_report(y_test, y_pred)}")

    # Validação cruzada
    cross_val_scores = cross_val_score(best_model, X_train, y_train, cv=5, scoring='f1_weighted')
    logger.info(f"Validação cruzada F1-scores: {cross_val_scores}")
    logger.info(f"F1-score médio (Validação Cruzada): {np.mean(cross_val_scores):.2f}")

    # Curva ROC e ajuste do threshold
    classes = ['negativo', 'neutro', 'positivo']
    y_test_bin = label_binarize(y_test, classes=classes)
    y_proba = best_model.predict_proba(X_test)
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    thresholds = dict()
    for i in range(len(classes)):
        fpr[i], tpr[i], thresholds[i] = roc_curve(y_test_bin[:, i], y_proba[:, i])
        roc_auc[i] = roc_auc_score(y_test_bin[:, i], y_proba[:, i])
    fig_path = config['directories']['figures'] + 'roc_curve.png'
    plot_roc_curve(fpr, tpr, roc_auc, classes, fig_path)
    logger.info(f"Área sob a curva ROC para cada classe: {roc_auc}")

    # Encontrar o melhor threshold
    best_thresholds = dict()
    for i in range(len(classes)):
        J = tpr[i] - fpr[i]
        ix = np.argmax(J)
        best_thresholds[classes[i]] = thresholds[i][ix]
    logger.info(f"Melhores thresholds por classe: {best_thresholds}")

    # Aplicar o melhor threshold
    y_pred_best_threshold = np.zeros_like(y_test_bin)
    for i in range(len(classes)):
        y_pred_best_threshold[:, i] = (y_proba[:, i] >= best_thresholds[classes[i]]).astype(int)
    y_pred_final = [classes[np.argmax(pred)] for pred in y_pred_best_threshold]
    report_best_threshold = classification_report(y_test, y_pred_final, output_dict=True)
    logger.info(f"Relatório de classificação (Melhor Threshold):\n {classification_report(y_test, y_pred_final)}")

    # Salvar o melhor modelo treinado
    model_path = config['directories']['models'] + 'melhor_modelo_rf.joblib'
    joblib.dump(best_model, model_path)
    logger.info("Melhor modelo salvo em %s", model_path)

if __name__ == "__main__":
    modelar_dados()
