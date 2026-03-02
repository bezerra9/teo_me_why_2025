# %%
import pandas as pd
import mlflow

mlflow.set_tracking_uri('http://localhost:5000')
model = mlflow.sklearn.load_model('models:/model_churn/2')
#model_df = pd.read_pickle('model.pkl')
#model_df -> posso fazer de outra maneira usando mlflow

# %%
features = model.feature_names_in_
features
# %%
model

# %%
df = pd.read_csv('../data/abt_churn.csv')
amostra = df[df['dtRef'] == df['dtRef'].max()].sample(3)
amostra
amostra = amostra.drop('flagChurn', axis=1)

# %%
predicao = model.predict_proba(amostra[features])[:,1]
amostra['proba_new'] = predicao
amostra
# %%
