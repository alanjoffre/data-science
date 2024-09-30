import pandas as pd
from datetime import datetime
import numpy as np
import random as python_random
import joblib

from sklearn.preprocessing import StandardScaler,LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE
import tensorflow as tf

from utils import *
import const

#reprodutividade
seed = 41
np.random.seed(seed)
python_random.seed(seed)
tf.random.set_seed(seed)

#dados brutos
df = fetch_data_from_db(const.consulta_sql)

#conversão de tipo
df['idade'] = df['idade'].astype(int)
df['valorsolicitado'] = df['valorsolicitado'].astype(float)
df['valortotalbem'] = df['valortotalbem'].astype(float)

#Tratamento de Nulos
substitui_nulos(df)

#Trata Erros de Digitação
profissoes_validas = ['Advogado', 'Arquiteto', 'Cientista de Dados', 'Contador','Dentista','Empresário',
                                       'Engenheiro','Médico','Programador']
corrigir_erros_digitacao(df, 'profissao', profissoes_validas)

#Trata Outliers
df = tratar_outliers(df, 'tempoprofissao', 0, 70)
df = tratar_outliers(df, 'idade', 0, 110)

#Feature Engineering
df['proporcaosolicitadototal'] = df['valorsolicitado'] / df['valortotalbem']
df['proporcaosolicitadototal'] = df['proporcaosolicitadototal'].astype(float)

# Dividindo Dados
X = df.drop('classe', axis=1)
y = df['classe']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, 
                                                    random_state=seed)

# Normalização
X_test = save_scalers(X_test, ['tempoprofissao','renda','idade',
                               'dependentes','valorsolicitado','valortotalbem','proporcaosolicitadototal'])
X_train = save_scalers(X_train, ['tempoprofissao','renda','idade',
                                 'dependentes','valorsolicitado','valortotalbem','proporcaosolicitadototal'])

#Codificação
mapeamento = {'ruim': 0, 'bom': 1}
y_train = np.array([mapeamento[item] for item in y_train])
y_test = np.array([mapeamento[item] for item in y_test])
X_train = save_encoders(X_train, ['profissao', 'tiporesidencia', 
                                  'escolaridade','score','estadocivil','produto'])
X_test = save_encoders(X_test, ['profissao', 'tiporesidencia', 
                                'escolaridade','score','estadocivil','produto'])

# Seleção de Atributos
model = RandomForestClassifier()
# Instancia o RFE
selector = RFE(model, n_features_to_select=10, step=1)
selector = selector.fit(X_train, y_train)
# Transforma os dados
X_train = selector.transform(X_train)
X_test = selector.transform(X_test)
joblib.dump(selector, './objects/selector.joblib')