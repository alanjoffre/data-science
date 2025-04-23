import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import optuna
import joblib
import os

# Função para avaliar o modelo
def avaliar_modelo(model, X_test, y_test):
    y_pred = model.predict(X_test)
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    return rmse, mae, r2

# Função de carregamento e pré-processamento de dados com amostra
def carregar_e_pre_processar_dados(caminho, amostra_frac=1.0):
    df = pd.read_parquet(caminho)

    # Obter uma amostra do dataset, se especificado
    if amostra_frac < 1.0:
        df = df.sample(frac=amostra_frac, random_state=42).reset_index(drop=True)

    df['data_inicio'] = pd.to_datetime(df['data_inicio'])
    df['data_final'] = pd.to_datetime(df['data_final'])
    df['ano_inicio'] = df['data_inicio'].dt.year
    df['mes_inicio'] = df['data_inicio'].dt.month
    df['dia_inicio'] = df['data_inicio'].dt.day
    df['ano_final'] = df['data_final'].dt.year
    df['mes_final'] = df['data_final'].dt.month
    df['dia_final'] = df['data_final'].dt.day

    # Adicionar a coluna 'viagens_agrupadas'
    df['viagens_agrupadas'] = df['ano_inicio'].astype(str) + '-' + df['mes_inicio'].astype(str) + '-' + df['dia_inicio'].astype(str)

    numerical_cols = ['segundos_da_viagem', 'milhas_da_viagem', 'area_comunitaria_do_embarque', 
                      'area_comunitaria_do_desembarque', 'tarifa', 'cobrancas_adicionais', 
                      'total_da_viagem', 'latitude_do_centroide_do_embarque', 'longitude_do_centroide_do_embarque', 
                      'latitude_do_centroide_do_desembarque', 'longitude_do_centroide_do_desembarque']
    categorical_cols = ['local_do_centroide_do_embarque', 'local_do_centroide_do_desembarque']

    df[numerical_cols] = df[numerical_cols].fillna(df[numerical_cols].mean())
    df[categorical_cols] = df[categorical_cols].fillna(df[categorical_cols].mode().iloc[0])
    df['trato_do_censo_do_embarque'] = df['trato_do_censo_do_embarque'].fillna(df['trato_do_censo_do_embarque'].median())
    df['trato_do_censo_do_desembarque'] = df['trato_do_censo_do_desembarque'].fillna(df['trato_do_censo_do_desembarque'].median())

    df['hora_dia_inicio'] = df['data_inicio'].dt.hour
    df['dia_semana_inicio'] = df['data_inicio'].dt.dayofweek

    X = df.drop(columns=['gorjeta', 'data_inicio', 'data_final'])
    y = df['gorjeta']

    numeric_transformer = Pipeline(steps=[('scaler', StandardScaler())])

    categorical_transformer = Pipeline(steps=[('onehot', OneHotEncoder(handle_unknown='ignore'))])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, X.select_dtypes(include=[np.number]).columns.tolist()),
            ('cat', categorical_transformer, categorical_cols)
        ])

    return preprocessor, X, y

# Carregar e pré-processar os dados
dataset_path = "D:/Github/data-science/projetos/logistica_transporte/1_previsao_gorjeta_regressao/data/processed/dataset_processado.parquet"
preprocessor, X, y = carregar_e_pre_processar_dados(dataset_path, amostra_frac=0.05)  # Usando 5% da amostra

# Dividir o dataset em conjunto de treinamento e teste (70/30)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Definir a função objetivo para o Optuna
def objective(trial):
    param = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 500),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'min_samples_split': trial.suggest_int('min_samples_split', 2, 20)
    }
    model = GradientBoostingRegressor(**param, random_state=42)
    
    pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('model', model)])
    
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(pipeline, X_train, y_train, cv=kf, scoring='neg_mean_squared_error')
    rmse = np.sqrt(-scores.mean())
    return rmse

# Realizar a busca de hiperparâmetros com Optuna
study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=5)  # 5 trials para otimização

# Aplicar os melhores hiperparâmetros
best_params = study.best_params
best_model = GradientBoostingRegressor(**best_params, random_state=42)

pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('model', best_model)])

# Treinar o pipeline
pipeline.fit(X_train, y_train)

# Avaliar o modelo nos dados de treino
train_rmse, train_mae, train_r2 = avaliar_modelo(pipeline, X_train, y_train)
print(f"Desempenho nos dados de treino - RMSE: {train_rmse}, MAE: {train_mae}, R²: {train_r2}")

# Avaliar o modelo nos dados de teste
test_rmse, test_mae, test_r2 = avaliar_modelo(pipeline, X_test, y_test)
print(f"Desempenho nos dados de teste - RMSE: {test_rmse}, MAE: {test_mae}, R²: {test_r2}")

# Realizar validação cruzada nos dados de treino
kf = KFold(n_splits=5, shuffle=True, random_state=42)
cross_val_scores = cross_val_score(pipeline, X_train, y_train, cv=kf, scoring='neg_mean_squared_error')
cross_val_rmse = np.sqrt(-cross_val_scores)
print(f"Validação cruzada nos dados de treino - RMSE: {cross_val_rmse.mean()}, STD: {cross_val_rmse.std()}")

# Salvar o pipeline
model_save_path = "D:/Github/data-science/projetos/logistica_transporte/1_previsao_gorjeta_regressao/models/pipeline_model.pkl"
os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
joblib.dump(pipeline, model_save_path)

print("Modelo e pipeline salvos com sucesso.")
