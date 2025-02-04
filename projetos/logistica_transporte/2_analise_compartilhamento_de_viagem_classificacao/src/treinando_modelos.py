import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from scipy import stats
from sklearn.ensemble import IsolationForest, RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report
from xgboost import XGBClassifier

# ---------------------- Carregamento e Processamento ---------------------- #
caminho_parquet = r'D:\Github\data-science\projetos\logistica_transporte\2_analise_compartilhamento_de_viagem_classificacao\data\processed\processado.parquet'
df = pd.read_parquet(caminho_parquet)
total_linhas = df.shape[0]
print("Total de linhas importadas:", total_linhas)
print("\nColunas e tipos de dados:")
print(df.dtypes)

# (As etapas de imputação, remoção de outliers, feature engineering, padronização já foram realizadas no pipeline anterior)

# ---------------------- Separação de Features e Target ---------------------- #
# Definindo a variável alvo
target = 'viagem_compartilhada_autorizada'
# Remover colunas que não serão usadas como features (identificadores, datas e localizações)
colunas_excluir = ['id_da_viagem', 'data_hora_de_inicio_da_viagem', 
                   'data_hora_de_termino_da_viagem', 
                   'localizacao_centro_de_partida', 'localizacao_centro_de_destino']

X = df.drop(columns=colunas_excluir + [target])
y = df[target].astype(int)  # Converte para 0/1 se estiver em bool

print("\nFeatures utilizadas:", X.columns.tolist())

# ---------------------- Divisão em Treino e Teste ---------------------- #
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print("\nTamanho do treino:", X_train.shape)
print("Tamanho do teste:", X_test.shape)

# ---------------------- Modelagem e Avaliação ---------------------- #
modelos = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, random_state=42),
    "Support Vector Classifier": SVC(probability=True, random_state=42),
    "Extra Trees": ExtraTreesClassifier(n_estimators=100, random_state=42),
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
resultados = {}

print("\nAvaliação com validação cruzada (F1-score médio):")
for nome_modelo, modelo in modelos.items():
    try:
        scores = cross_val_score(modelo, X_train, y_train, cv=cv, scoring='f1')
        resultados[nome_modelo] = np.mean(scores)
        print(f"{nome_modelo}: F1-score médio = {np.mean(scores):.4f}")
    except Exception as e:
        print(f"{nome_modelo}: Erro durante cross_val_score: {e}")

melhor_modelo_nome = max(resultados, key=resultados.get)
melhor_modelo = modelos[melhor_modelo_nome]
print(f"\nMelhor modelo na validação cruzada: {melhor_modelo_nome} com F1-score médio = {resultados[melhor_modelo_nome]:.4f}")

# Treinar o melhor modelo e avaliar no conjunto de teste
melhor_modelo.fit(X_train, y_train)
y_pred = melhor_modelo.predict(X_test)
if hasattr(melhor_modelo, "predict_proba"):
    y_proba = melhor_modelo.predict_proba(X_test)[:, 1]
else:
    y_proba = None

acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_proba) if y_proba is not None else None

print("\nMétricas no conjunto de teste:")
print(f"Acurácia: {acc:.4f}")
print(f"Precisão: {prec:.4f}")
print(f"Recall: {rec:.4f}")
print(f"F1-score: {f1:.4f}")
if auc is not None:
    print(f"ROC AUC: {auc:.4f}")

print("\nRelatório de Classificação:")
print(classification_report(y_test, y_pred))

print(f"\nO modelo com melhor desempenho foi: {melhor_modelo_nome} com F1-score médio = {resultados[melhor_modelo_nome]:.4f}")
