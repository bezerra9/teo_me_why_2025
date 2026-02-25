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
#feature_importances[feature_importances[0]>0.01] # ou
feature_importances[feature_importances['acum.'] < 0.96]

# Explicação -> usamos o `arvore.feature_importances_` para verificar quais variaveis contribuiram mais com as 'quebras' da minha arvore, transformamos isso em uma series e ordenamos em ordem crescente, apos isso aplicamos o metodo `cumsum()` que vai acumular as features_importances da arvore e vai nos dar até mais ou menos que ponto as variaveis utilizadas pela arvore durante a quebra podem nos ajudar no nosso modelo final. Não sei até que ponto isso se difere de uma plotagem de heatmap para verificar a correlação de algumas variaveis


# %%
best_features = (feature_importances[feature_importances['acum.'] < 0.96]['index'].tolist())
best_features


# %%
# MODIFY
from feature_engine import discretisation

tree_discretization = discretisation.DecisionTreeDiscretiser(variables=best_features,       regression=False, bin_output='bin_number', cv=3)
#Estou usando arvore para criar os bins no meu dataset
X_train.head()

tree_discretization.fit(X_train[best_features], y_train)
# %%
X_train_transform = tree_discretization.transform(X_train[best_features])

X_train_transform


# %%
# Model
from sklearn import linear_model

reg = linear_model.LogisticRegression(penalty=None, max_iter=1000000, random_state=42)
reg.fit(X_train_transform, y_train)


# %%
from sklearn import metrics

y_train_predict = reg.predict(X_train_transform)
y_train_proba = reg.predict_proba(X_train_transform)[:, 1]

acc_train = metrics.accuracy_score(y_train, y_train_predict)
auc_train = metrics.roc_auc_score(y_train, y_train_proba)
print('Acuracia Treino: ', acc_train)
print('AUC Treino: ', auc_train)


# %%
X_test_transform = tree_discretization.transform(X_test[best_features])
X_test_transform

y_test_predict = reg.predict(X_test_transform)
y_test_proba = reg.predict_proba(X_test_transform)[:, 1]

acc_test = metrics.accuracy_score(y_test, y_test_predict)
auc_test = metrics.roc_auc_score(y_test, y_test_proba)
print('Acuracia teste: ', acc_test)
print('AUC teste: ', auc_test)


# %%
oot_transform = tree_discretization.transform(oot[best_features])


y_oot_predict = reg.predict(oot_transform)
y_oot_proba = reg.predict_proba(oot_transform)[:, 1]

acc_oot = metrics.accuracy_score(oot[target], y_oot_predict)
auc_oot = metrics.roc_auc_score(oot[target], y_oot_proba)
print('Acuracia oot: ', acc_oot)
print('AUC oot: ', auc_oot)


# %%
