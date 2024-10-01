import os
import pandas as pd
import numpy as np
import random as python_random
import joblib
import tensorflow as tf
import logging
from datetime import datetime
import time
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
import xgboost as xgb
import lightgbm as lgb
from sklearn.feature_selection import RFE
from imblearn.over_sampling import SMOTE  # Para lidar com desbalanceamento
from utils import *
import const

# Configuração de logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def set_seed(seed=41):
    """Define a semente para reprodutibilidade."""
    np.random.seed(seed)
    python_random.seed(seed)
    tf.random.set_seed(seed)

def load_data():
    """Carrega dados da base de dados."""
    try:
        start_time = time.time()
        df = fetch_data_from_db(const.consulta_sql)
        elapsed_time = time.time() - start_time
        logging.info(f"Dados carregados em {elapsed_time:.2f} segundos.")
        return df
    except Exception as e:
        logging.error(f"Erro ao carregar dados: {e}")
        raise

def preprocess_data(df):
    """Realiza o pré-processamento dos dados."""
    df['idade'] = df['idade'].astype(int)
    df['valorsolicitado'] = df['valorsolicitado'].astype(float)
    df['valortotalbem'] = df['valortotalbem'].astype(float)

    substitui_nulos(df)

    profissoes_validas = ['Advogado', 'Arquiteto', 'Cientista de Dados', 'Contador', 'Dentista', 
                          'Empresário', 'Engenheiro', 'Médico', 'Programador']
    corrigir_erros_digitacao(df, 'profissao', profissoes_validas)

    df = tratar_outliers(df, 'tempoprofissao', 0, 70)
    df = tratar_outliers(df, 'idade', 0, 110)

    df['proporcaosolicitadototal'] = df['valorsolicitado'] / df['valortotalbem']
    df['proporcaosolicitadototal'] = df['proporcaosolicitadototal'].astype(float)

    return df

def split_data(df):
    """Divide os dados em conjuntos de treino e teste."""
    X = df.drop('classe', axis=1)
    y = df['classe']
    return train_test_split(X, y, test_size=0.2, random_state=41, stratify=y)

def normalize_and_encode(X_train, X_test, y_train, y_test):
    """Normaliza e codifica os dados."""
    features_to_scale = ['tempoprofissao', 'renda', 'idade', 'dependentes', 
                         'valorsolicitado', 'valortotalbem', 'proporcaosolicitadototal']
    
    X_train_scaled, X_test_scaled = save_scalers(X_train, features_to_scale), save_scalers(X_test, features_to_scale)

    # Codificação das variáveis de saída
    mapeamento = {'ruim': 0, 'bom': 1}
    y_train_encoded = np.array([mapeamento[item] for item in y_train])
    y_test_encoded = np.array([mapeamento[item] for item in y_test])

    # Codificação de variáveis categóricas
    categorical_features = ['profissao', 'tiporesidencia', 'escolaridade', 
                            'score', 'estadocivil', 'produto']
    X_train_encoded, X_test_encoded = save_encoders(X_train_scaled, categorical_features), save_encoders(X_test_scaled, categorical_features)

    return X_train_encoded, X_test_encoded, y_train_encoded, y_test_encoded

def handle_imbalanced_data(X_train, y_train):
    """Lida com dados desbalanceados usando SMOTE."""
    smote = SMOTE(random_state=41)
    X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
    return X_resampled, y_resampled

def select_features(X_train, y_train, X_test):
    """Seleciona características relevantes utilizando RFE."""
    model_rf = RandomForestClassifier(random_state=41)
    selector = RFE(model_rf, n_features_to_select=10, step=1)
    selector.fit(X_train, y_train)
    X_train_selected = selector.transform(X_train)
    X_test_selected = selector.transform(X_test)  # Aplica a seleção no conjunto de teste
    return X_train_selected, X_test_selected, selector

def save_model(model, model_name):
    """Salva o modelo treinado em um arquivo usando joblib."""
    try:
        joblib.dump(model, f"{model_name}.joblib")
        logging.info(f"Modelo {model_name} salvo com sucesso.")
    except Exception as e:
        logging.error(f"Erro ao salvar o modelo {model_name}: {e}")

def load_model(model_name):
    """Carrega o modelo treinado de um arquivo usando joblib."""
    try:
        model = joblib.load(f"{model_name}.joblib")
        logging.info(f"Modelo {model_name} carregado com sucesso.")
        return model
    except FileNotFoundError:
        logging.error(f"Modelo {model_name} não encontrado.")
        raise
    except Exception as e:
        logging.error(f"Erro ao carregar o modelo {model_name}: {e}")
        raise

def train_models(X_train, y_train, X_test, y_test):
    """Treina diferentes modelos e avalia seu desempenho."""
    modelos = {
        'RandomForest': RandomForestClassifier(random_state=41),
        'GradientBoosting': GradientBoostingClassifier(random_state=41),
        'SVM': SVC(kernel='linear', probability=True, random_state=41),
        'KNN': KNeighborsClassifier(),
        'LogisticRegression': LogisticRegression(random_state=41),
        'DecisionTree': DecisionTreeClassifier(random_state=41),
        'XGBoost': xgb.XGBClassifier(random_state=41, use_label_encoder=False, eval_metric='logloss'),
        'AdaBoost': AdaBoostClassifier(n_estimators=100, random_state=41),
        'LightGBM': lgb.LGBMClassifier(random_state=41)
    }

    resultados = {}
    melhor_acuracia = 0
    melhor_modelo = None

    for nome_modelo, modelo in modelos.items():
        logging.info(f"Iniciando treinamento do modelo: {nome_modelo}")
        try:
            # Avaliação com validação cruzada
            scores = cross_val_score(modelo, X_train, y_train, cv=5)
            modelo.fit(X_train, y_train)
            y_pred = modelo.predict(X_test)
            acuracia = np.mean(scores)

            # Cálculo de métricas adicionais
            report = classification_report(y_test, y_pred, output_dict=True)
            f1_score = report['1']['f1-score']
            precision = report['1']['precision']
            recall = report['1']['recall']

            resultados[nome_modelo] = {
                'accuracy': acuracia,
                'f1_score': f1_score,
                'precision': precision,
                'recall': recall,
                'classification_report': report,
                'y_pred': y_pred
            }

            logging.info(f"{nome_modelo} - Acurácia: {acuracia:.4f}, F1 Score: {f1_score:.4f}, Precisão: {precision:.4f}, Recall: {recall:.4f}")

            if acuracia > melhor_acuracia:
                melhor_acuracia = acuracia
                melhor_modelo = {
                    'modelo': modelo,
                    'nome': nome_modelo,
                    'classification_report': report,
                    'accuracy': acuracia,
                    'f1_score': f1_score,
                    'precision': precision,
                    'recall': recall
                }
                save_model(modelo, nome_modelo)  # Salvar o melhor modelo treinado

        except Exception as e:
            logging.error(f"Erro ao treinar o modelo {nome_modelo}: {e}")

    return resultados, melhor_modelo

def train_neural_network(X_train, y_train):
    """Treina uma rede neural para classificação."""
    model_nn = tf.keras.Sequential([
        tf.keras.layers.Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    model_nn.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    
    # Treinamento
    history = model_nn.fit(X_train, y_train, epochs=100, batch_size=32, validation_split=0.2, verbose=0)
    
    save_model(model_nn, 'neural_network')  # Salvar a rede neural treinada
    logging.info("Rede Neural treinada e salva com sucesso.")

    return history

def main():
    """Função principal do programa."""
    set_seed()

    df = load_data()
    df = preprocess_data(df)
    
    X_train, X_test, y_train, y_test = split_data(df)
    X_train_encoded, X_test_encoded, y_train_encoded, y_test_encoded = normalize_and_encode(X_train, X_test, y_train, y_test)

    # Lida com dados desbalanceados
    X_train_resampled, y_train_resampled = handle_imbalanced_data(X_train_encoded, y_train_encoded)

    # Seleciona características relevantes
    X_train_selected, X_test_selected, selector = select_features(X_train_resampled, y_train_resampled, X_test_encoded)

    # Treina modelos
    resultados, melhor_modelo = train_models(X_train_selected, y_train_resampled, X_test_selected, y_test_encoded)

    # Treina a rede neural
    history_nn = train_neural_network(X_train_selected, y_train_resampled)

    # Carrega o modelo treinado, se necessário
    try:
        model_nn = load_model('neural_network')  # Carregar a rede neural
    except FileNotFoundError:
        logging.error("Modelo de rede neural não encontrado, verifique se foi treinado.")

    # Informar melhor modelo e suas métricas
    if melhor_modelo is not None:
        logging.info(f"\nMelhor Modelo: {melhor_modelo['nome']}")
        logging.info(f"Acurácia: {melhor_modelo['accuracy']:.4f}")
        logging.info("Relatório de Classificação:")
        logging.info(f"{melhor_modelo['classification_report']}")

if __name__ == "__main__":
    main()
