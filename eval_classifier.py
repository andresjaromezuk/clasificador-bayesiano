# %%
import mlflow
import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis, LinearDiscriminantAnalysis

mlflow.set_experiment("Bayesian classifier")

df_train = pd.read_csv("./data/alturas-pesos-mils-train.csv")
df_test  = pd.read_csv("./data/alturas-pesos-mils-test.csv")
# %%
model_list = [
  {'model': GaussianNB, 'model_name': 'GaussianNB'},
  {'model': QuadraticDiscriminantAnalysis, 'model_name': 'QuadraticDiscriminantAnalysis'},
  {'model': LinearDiscriminantAnalysis, 'model_name': 'LinearDiscriminantAnalysis'},
]
model_selected = model_list[1]
model = model_selected['model'](reg_param=0.8)
mlflow.start_run(run_name=model_selected['model_name'])

mlflow.log_params({'reg_param': 0.8})


# %%
#entrenamiento modelo
input_train = df_train[['Peso', 'Altura']]
output_train = df_train[['Genero']]

input_test = df_test[['Peso', 'Altura']]
output_test = df_test[['Genero']]

model.fit(input_train, output_train)

# %%
#model.classes_

#NaiveBayes
#model.class_count_ #conteo
#model.theta_ #medias
#model.var_ #varianzas

#QDA
#model.means_
#model.covariance_

# %%
#predicción
#model.predict([[50, 165]])

#probabilidades
#model.predict_proba([[50, 160]])

#accuracy
train_accuracy = model.score(input_train, output_train)
test_accuracy = model.score(input_test, output_test)

mlflow.log_metrics({'train_accuracy': train_accuracy,'test_accuracy':test_accuracy})

mlflow.end_run()



# %%
