#!/usr/bin/env python
import os
import json
import logging
import yaml
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.model_selection import (train_test_split, GridSearchCV, 
                                     StratifiedKFold, cross_val_score)
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, classification_report,
                             roc_curve, precision_recall_curve)
from sklearn.calibration import CalibratedClassifierCV
import joblib

# ---------------------- Carregar Configurações do YAML ---------------------- #
CONFIG_PATH = r'D:\Github\data-science\projetos\logistica_transporte\2_analise_compartilhamento_de_viagem_classificacao\config\config.yaml'
with open(CONFIG_PATH, 'r') as file:
    config = yaml.safe_load(file)

# Extraindo os caminhos do config.yaml
BASE_DIR = r'D:\Github\data-science\projetos\logistica_transporte\2_analise_compartilhamento_de_viagem_classificacao'
LOGS_DIR = config['paths']['logs_dir']
MODELS_DIR = config['models']['directory']
PREPROCESSORS_DIR = os.path.dirname(config['preprocessors']['path'])
HYPERPARAMETERS_DIR = r'D:\Github\data-science\projetos\logistica_transporte\2_analise_compartilhamento_de_viagem_classificacao\hyperparameters'
METRICS_DIR = config['reports']['directory']
FIGURES_DIR = config['reports']['figures_dir']
DATA_PROCESSED = config['data']['processed']

# Garantir que os diretórios existam
for d in [LOGS_DIR, MODELS_DIR, HYPERPARAMETERS_DIR, METRICS_DIR, FIGURES_DIR]:
    os.makedirs(d, exist_ok=True)

# Configuração do logging
logging.basicConfig(
    filename=os.path.join(LOGS_DIR, 'modelo_random_forest.log'),
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# ---------------------- Funções Auxiliares ---------------------- #
def report_data_info(df):
    total_linhas = df.shape[0]
    logging.info(f"Número de linhas importadas: {total_linhas}")
    print("Número de linhas importadas:", total_linhas)
    
    logging.info("Colunas e tipos de dados:")
    logging.info(df.dtypes.to_dict())
    print("\nColunas e tipos de dados:")
    print(df.dtypes)
    
    logging.info("Lista de colunas:")
    logging.info(df.columns.tolist())
    print("\nLista de colunas:")
    print(df.columns.tolist())
    
    # Verificação de dados faltantes
    print("\nVerificação de dados faltantes:")
    missing = df.isna().sum()
    for col, count in missing.items():
        percent = (count / total_linhas) * 100
        logging.info(f"{col}: {count} faltantes, {percent:.2f}%")
        print(f"{col}: {count} valores faltantes, {percent:.2f}% do total de linhas importadas")
    
    # Dados duplicados
    dup = df.duplicated().sum()
    dup_percent = (dup / total_linhas) * 100
    logging.info(f"Linhas duplicadas: {dup} ({dup_percent:.2f}%)")
    print("\nVerificação de dados duplicados:")
    print(f"Número de linhas duplicadas: {dup}, {dup_percent:.2f}% do total de linhas importadas")
    
    # Outliers (usando Z-score para colunas numéricas)
    print("\nVerificação de outliers (colunas numéricas) usando Z-score:")
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    for col in numeric_cols:
        z_scores = np.abs(stats.zscore(df[col].dropna()))
        out_count = np.sum(z_scores > 3)
        out_percent = (out_count / total_linhas) * 100
        logging.info(f"{col}: {out_count} outliers, {out_percent:.2f}%")
        print(f"{col}: {out_count} outliers, {out_percent:.2f}% do total de linhas importadas")

def nested_cv_tuning(X, y, cv_outer=5, cv_inner=5):
    """
    Executa Nested Cross-Validation: o outer loop estima a performance e o inner realiza o tuning.
    Retorna o melhor modelo treinado com os melhores hiperparâmetros encontrados e a média do F1-score do outer loop.
    """
    # Converter y para array NumPy para garantir indexação correta
    if isinstance(y, pd.Series):
        y = y.values

    param_grid = {
        'clf__n_estimators': [100, 200],
        'clf__max_depth': [None, 10, 20],
        'clf__min_samples_split': [2, 5]
    }
    outer_cv = StratifiedKFold(n_splits=cv_outer, shuffle=True, random_state=42)
    outer_scores = []
    best_models = []
    
    for train_ix, test_ix in outer_cv.split(X, y):
        X_train_outer, X_test_outer = X[train_ix], X[test_ix]
        y_train_outer, y_test_outer = y[train_ix], y[test_ix]
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('clf', RandomForestClassifier(random_state=42))
        ])
        inner_cv = StratifiedKFold(n_splits=cv_inner, shuffle=True, random_state=42)
        grid_search = GridSearchCV(pipeline, param_grid, scoring='f1', cv=inner_cv, n_jobs=-1, verbose=0)
        grid_search.fit(X_train_outer, y_train_outer)
        best_model = grid_search.best_estimator_
        y_pred_outer = best_model.predict(X_test_outer)
        score = f1_score(y_test_outer, y_pred_outer)
        outer_scores.append(score)
        best_models.append(best_model)
    avg_score = np.mean(outer_scores)
    best_index = np.argmax(outer_scores)
    best_overall_model = best_models[best_index]
    logging.info(f"Nested CV F1-score médio: {avg_score:.4f}")
    print(f"Nested CV F1-score médio: {avg_score:.4f}")
    best_params = best_overall_model.named_steps['clf'].get_params()
    return best_overall_model, best_params, avg_score

def tune_threshold(model, X, y, thresholds=np.linspace(0.1, 0.9, 81)):
    from sklearn.metrics import f1_score
    if not hasattr(model, "predict_proba"):
        raise ValueError("O modelo não possui predict_proba")
    y_proba = model.predict_proba(X)[:, 1]
    best_thresh = 0.5
    best_f1 = 0
    for thresh in thresholds:
        y_pred_thresh = (y_proba >= thresh).astype(int)
        current_f1 = f1_score(y, y_pred_thresh)
        if current_f1 > best_f1:
            best_f1 = current_f1
            best_thresh = thresh
    logging.info(f"Melhor threshold: {best_thresh} com F1-score = {best_f1:.4f}")
    print(f"Melhor threshold: {best_thresh} com F1-score = {best_f1:.4f}")
    return best_thresh

def save_roc_curve(model, X_test, y_test, filename):
    from sklearn.metrics import roc_curve, roc_auc_score
    if not hasattr(model, "predict_proba"):
        return
    y_proba = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    plt.figure()
    plt.plot(fpr, tpr, label='ROC curve (AUC = %0.4f)' % roc_auc_score(y_test, y_proba))
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Curva ROC')
    plt.legend(loc="lower right")
    plt.savefig(filename)
    plt.close()
    logging.info("Curva ROC salva em: " + filename)

def save_precision_recall_curve(model, X_test, y_test, filename):
    from sklearn.metrics import precision_recall_curve
    if not hasattr(model, "predict_proba"):
        return
    y_proba = model.predict_proba(X_test)[:, 1]
    precision_vals, recall_vals, _ = precision_recall_curve(y_test, y_proba)
    plt.figure()
    plt.plot(recall_vals, precision_vals, label='Precision-Recall curve')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Curva Precision-Recall')
    plt.legend(loc="lower left")
    plt.savefig(filename)
    plt.close()
    logging.info("Curva Precision-Recall salva em: " + filename)

# ---------------------- Programa Principal ---------------------- #
def main():
    # Carregar dataset e gerar relatório inicial
    df = pd.read_parquet(config['data']['processed'])
    total_linhas = df.shape[0]
    print("Número de linhas importadas:", total_linhas)
    report_data_info(df)
    
    # Preparação dos dados para modelagem: remover colunas irrelevantes
    colunas_excluir = ['id_da_viagem', 'data_hora_de_inicio_da_viagem', 
                       'data_hora_de_termino_da_viagem',
                       'localizacao_centro_de_partida', 'localizacao_centro_de_destino']
    target = 'viagem_compartilhada_autorizada'
    X = df.drop(columns=colunas_excluir + [target])
    y = df[target].astype(int)
    print("\nFeatures utilizadas para modelagem:")
    print(X.columns.tolist())
    
    # Divisão dos dados: 70% treino, 30% teste (random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    print("\nTamanho do treino:", X_train.shape)
    print("Tamanho do teste:", X_test.shape)
    
    # Pré-processamento: escalonamento (Pipeline) e salvamento do pré-processador
    preprocessor = Pipeline([('scaler', StandardScaler())])
    X_train = preprocessor.fit_transform(X_train)
    X_test = preprocessor.transform(X_test)
    preprocessor_path = os.path.join(PREPROCESSORS_DIR, 'preprocessor.joblib')
    joblib.dump(preprocessor, preprocessor_path)
    logging.info("Pré-processador salvo em: " + preprocessor_path)
    
    # Nested Cross-Validation para tuning de hiperparâmetros
    print("\nIniciando Nested Cross-Validation para tuning de hiperparâmetros...")
    best_pipeline, best_params, nested_cv_score = nested_cv_tuning(X_train, y_train, cv_outer=5, cv_inner=5)
    hyperparameters_path = os.path.join(HYPERPARAMETERS_DIR, 'best_hyperparameters.json')
    with open(hyperparameters_path, 'w') as f:
        json.dump(best_params, f, indent=4)
    logging.info("Hiperparâmetros salvos em: " + hyperparameters_path)
    
    # Model Calibration: Calibrar o modelo usando Platt Scaling (sigmoid)
    from sklearn.calibration import CalibratedClassifierCV
    # Extraindo o classificador do pipeline nested
    best_rf = best_pipeline.named_steps['clf']
    # Calibração com cv='prefit' pois o modelo já está ajustado
    calibrated_rf = CalibratedClassifierCV(base_estimator=best_rf, method='sigmoid', cv='prefit')
    calibrated_rf.fit(X_train, y_train)
    
    # Treinamento e avaliação no treino (usando o modelo calibrado)
    y_train_pred = calibrated_rf.predict(X_train)
    print("\nRelatório de classificação no TREINO:")
    print(classification_report(y_train, y_train_pred))
    
    # Avaliação no conjunto de teste
    y_test_pred = calibrated_rf.predict(X_test)
    print("\nRelatório de classificação no TESTE:")
    print(classification_report(y_test, y_test_pred))
    
    # Validação cruzada adicional usando o pipeline não calibrado (pois calibrated_rf com cv='prefit' não pode ser re-ajustado)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(best_pipeline, X_train, y_train, cv=cv, scoring='f1')
    print("\nValidação cruzada no TREINO (F1-score médio):", np.mean(cv_scores))
    
    # Threshold Tuning
    best_thresh = tune_threshold(calibrated_rf, X_train, y_train)
    y_train_proba = calibrated_rf.predict_proba(X_train)[:, 1]
    y_train_thresh = (y_train_proba >= best_thresh).astype(int)
    print("\nRelatório de classificação no TREINO (com threshold ajustado):")
    print(classification_report(y_train, y_train_thresh))
    
    y_test_proba = calibrated_rf.predict_proba(X_test)[:, 1]
    y_test_thresh = (y_test_proba >= best_thresh).astype(int)
    print("\nRelatório de classificação no TESTE (com threshold ajustado):")
    print(classification_report(y_test, y_test_thresh))
    
    # Salvar métricas
    metrics = {
        "Train F1-score (threshold adjusted)": f1_score(y_train, y_train_thresh),
        "Test F1-score (threshold adjusted)": f1_score(y_test, y_test_thresh),
        "Train Accuracy": accuracy_score(y_train, y_train_pred),
        "Test Accuracy": accuracy_score(y_test, y_test_pred),
        "Train Precision": precision_score(y_train, y_train_thresh),
        "Test Precision": precision_score(y_test, y_test_thresh),
        "Train Recall": recall_score(y_train, y_train_thresh),
        "Test Recall": recall_score(y_test, y_test_thresh),
        "Nested CV F1-score": nested_cv_score
    }
    metrics_path = os.path.join(METRICS_DIR, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=4)
    logging.info("Métricas salvas em: " + metrics_path)
    
    # Salvar o modelo final (modelo calibrado)
    model_path = os.path.join(MODELS_DIR, "final_model.joblib")
    joblib.dump(calibrated_rf, model_path)
    logging.info("Modelo salvo em: " + model_path)
    
    # Gerar e salvar gráfico ROC
    from sklearn.metrics import roc_curve, roc_auc_score
    roc_path = os.path.join(FIGURES_DIR, "roc_curve.png")
    fpr, tpr, _ = roc_curve(y_test, y_test_proba)
    plt.figure()
    plt.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc_score(y_test, y_test_proba):.4f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Curva ROC')
    plt.legend(loc="lower right")
    plt.savefig(roc_path)
    plt.close()
    logging.info("Curva ROC salva em: " + roc_path)
    
    # Gerar e salvar gráfico Precision-Recall
    pr_path = os.path.join(FIGURES_DIR, "precision_recall_curve.png")
    from sklearn.metrics import precision_recall_curve
    precision_vals, recall_vals, _ = precision_recall_curve(y_test, y_test_proba)
    plt.figure()
    plt.plot(recall_vals, precision_vals, label='Precision-Recall curve')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Curva Precision-Recall')
    plt.legend(loc="lower left")
    plt.savefig(pr_path)
    plt.close()
    logging.info("Curva Precision-Recall salva em: " + pr_path)

if __name__ == '__main__':
    main()
