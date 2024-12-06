import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier, ExtraTreesClassifier, BaggingClassifier
from sklearn.naive_bayes import MultinomialNB, ComplementNB
from sklearn.linear_model import LogisticRegression, RidgeClassifier, SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, roc_curve, auc
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline
import xgboost as xgb
import lightgbm as lgb
import catboost as cb
import numpy as np
import matplotlib.pyplot as plt
import joblib
import json

# Carregar o dataset
file_path = r'D:\Github\data-science\projetos\analise-de-sentimentos\nlp\data\processed\final_data.parquet'
df = pd.read_parquet(file_path)

# Features e target
X = df['tweet']
y = df['sentimento_codificado']

# Dividir o dataset em conjunto de treinamento e teste (70/30)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Vetorização de textos
vectorizer = TfidfVectorizer()
X_train_vec = vectorizer.fit_transform(X_train).toarray()
X_test_vec = vectorizer.transform(X_test).toarray()

# Salvar o vetor de TfidfVectorizer
preprocessor_path = r'D:\Github\data-science\projetos\analise-de-sentimentos\nlp\preprocessors\tfidf_vectorizer.pkl'
joblib.dump(vectorizer, preprocessor_path)

# Lidar com dados desbalanceados
over = ADASYN(sampling_strategy='minority')
X_train_resampled, y_train_resampled = over.fit_resample(X_train_vec, y_train)

# Modelos de Machine Learning
models_ml = {
    'RandomForest': RandomForestClassifier(random_state=42),
    'ExtraTrees': ExtraTreesClassifier(random_state=42),
    'NaiveBayes': MultinomialNB(),
    'ComplementNB': ComplementNB(),
    'LogisticRegression': LogisticRegression(max_iter=1000, random_state=42),
    'RidgeClassifier': RidgeClassifier(),
    'KNN': KNeighborsClassifier(),
    'GradientBoosting': GradientBoostingClassifier(random_state=42),
    'AdaBoost': AdaBoostClassifier(random_state=42),
    'DecisionTree': DecisionTreeClassifier(random_state=42),
    'SVM': SVC(random_state=42, probability=True),
    'XGBoost': xgb.XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42),
    'LightGBM': lgb.LGBMClassifier(random_state=42),
    'CatBoost': cb.CatBoostClassifier(verbose=0, random_state=42),
    'Bagging': BaggingClassifier(base_estimator=DecisionTreeClassifier(), n_estimators=10, random_state=42),
    'SGD': SGDClassifier(max_iter=1000, random_state=42)
}

# Avaliar modelos de Machine Learning
results = {}
for model_name, model in models_ml.items():
    model.fit(X_train_resampled, y_train_resampled)
    y_pred = model.predict(X_test_vec)
    report = classification_report(y_test, y_pred, zero_division=0, output_dict=True)
    results[model_name] = report
    print(f"Classification Report for {model_name}:\n", classification_report(y_test, y_pred, zero_division=0))

# Determinar o melhor modelo baseado no F1-Score médio ponderado
best_model_name = max(results, key=lambda name: results[name]['weighted avg']['f1-score'])
best_model = models_ml[best_model_name]
print(f"\nO melhor modelo de machine learning é {best_model_name} com F1-Score: {results[best_model_name]['weighted avg']['f1-score']}")

# Avaliar o modelo no conjunto de teste
y_pred_test_best = best_model.predict(X_test_vec)
y_scores_best = best_model.predict_proba(X_test_vec)[:, 1] if hasattr(best_model, "predict_proba") else best_model.decision_function(X_test_vec)

# Encontrar o melhor threshold usando a curva ROC
fpr_best, tpr_best, roc_thresholds_best = roc_curve(y_test, y_scores_best, pos_label=1)
roc_auc_best = auc(fpr_best, tpr_best)

optimal_idx_best = np.argmax(tpr_best - fpr_best)
optimal_threshold_best = roc_thresholds_best[optimal_idx_best]
print("Melhor Threshold pela Curva ROC - Melhor Modelo:", optimal_threshold_best)

# Aplicar o melhor threshold
y_pred_best_threshold_best = (y_scores_best >= optimal_threshold_best).astype(int)
print("Relatório de Classificação - Melhor Threshold pela Curva ROC - Melhor Modelo:\n" + classification_report(y_test, y_pred_best_threshold_best, zero_division=0))

# Salvar o modelo treinado com o melhor threshold
joblib.dump(best_model, r'D:\Github\data-science\projetos\analise-de-sentimentos\nlp\models\best_model.pkl')

# Salvar os hiperparâmetros
params_path_best_model = r'D:\Github\data-science\projetos\analise-de-sentimentos\nlp\reports\best_model_params.json'
with open(params_path_best_model, 'w') as file:
    json.dump(best_model.get_params(), file)

# Plotar a curva ROC e salvar o gráfico
plt.figure()
plt.plot(fpr_best, tpr_best, color='darkorange', lw=2, label=f'Curva ROC (área = {roc_auc_best:0.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.scatter(fpr_best[optimal_idx_best], tpr_best[optimal_idx_best], marker='o', color='red', label='Melhor Threshold')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Taxa de Falsos Positivos')
plt.ylabel('Taxa de Verdadeiros Positivos')
plt.title('Curva ROC - Melhor Modelo de Machine Learning')
plt.legend(loc="lower right")

# Salvar o gráfico
plt.savefig(r'D:\Github\data-science\projetos\analise-de-sentimentos\nlp\reports\figures\roc_curve_best_model.png')
plt.close()
