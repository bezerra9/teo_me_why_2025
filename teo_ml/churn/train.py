# %%
import pandas as pd


df = pd.read_csv('../data/abt_churn.csv')
df.head()
# %%
df['dtRef'].sort_values().unique() #estamos observando para pegar uma parcela Out Of Time
df['dtRef'].value_counts().sort_index() #vou pegar a `2025-04-01` para ser a minha safra do Out Of Time
oot = df[df['dtRef']==df['dtRef'].max()].copy()

# %%
df_train = df[df['dtRef']<df['dtRef'].max()].copy()

# %%
features = df_train.columns[2:-1]
target = 'flagChurn'
X, y = df_train[features], df_train[target]
# %%
from sklearn import model_selection

X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, random_state=42, test_size=0.2, stratify=y)
# estratificamos para manter a mesma proporção das taxas da celula abaixo
# `estratificação` -> forma de garantir que as duas amostras tenham a mesma taxa da variavel resposta. Em algumas situações se precisar fazer balanceamento de base voce pode utilizar `oversampling` ou `undersampling` 

# %%
print('Taxa de variavel resposta Treino', y_train.mean())
print('Taxa de variavel resposta Teste', y_test.mean())