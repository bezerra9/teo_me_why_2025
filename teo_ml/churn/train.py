# %%
import pandas as pd

pd.options.display.max_columns = 500
pd.options.display.max_rows = 500

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
# SAMPLE
from sklearn import model_selection
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, random_state=42, test_size=0.2, stratify=y)
# estratificamos para manter a mesma proporção das taxas da celula abaixo
# `estratificação` -> forma de garantir que as duas amostras tenham a mesma taxa da variavel resposta. Em algumas situações se precisar fazer balanceamento de base voce pode utilizar `oversampling` ou `undersampling` 


# %%
print('Taxa de variavel resposta Treino', y_train.mean())
print('Taxa de variavel resposta Teste', y_test.mean())


# %%
# EXPLORE (MISSINGS)
X_train.isna().sum().sort_values(ascending=False)


# %%
df_analise = X_train.copy()
df_analise[target] = y_train
sumario = df_analise.groupby(by=target).agg(['mean', 'median']).T
sumario


# %%
sumario['diff_abs'] = sumario[0] - sumario[1]
sumario['diff_rel'] = sumario[0] / sumario[1]
sumario.sort_values(by=['diff_rel'], ascending=False)
# Média muito diferente da mediana é bom ficar de olho, é possivel que tenha uma distribuição esticada ou valores fora do comum


# %%
# Rodar uma árvore de decisão pode ser bom para explorar o dado (EXPLORE). Ajuda a entender quais variaveis estão nos ajudando
from sklearn import tree
import matplotlib.pyplot as plt

arvore = tree.DecisionTreeClassifier(random_state=42)
arvore.fit(X_train, y_train)


# %%
feature_importances = (pd.Series(arvore.feature_importances_, index=X_train.columns).sort_values(ascending=False).reset_index())

feature_importances['acum.'] = feature_importances[0].cumsum()
feature_importances[feature_importances[0]>0.01]