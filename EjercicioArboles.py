import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import plot_tree
from sklearn.tree import _tree
import matplotlib.pyplot as plt
import graphviz as gp


df = pd.read_csv("C:\\Users\\bauti\\OneDrive\\Escritorio\\Optativas\\Autonomia E Inteligencia Artificial\\Datos.csv")
df_4 = df[['Me_Gusta','Genero','Actores_Gustan','Duracion<2:30']]
df_4 = pd.get_dummies(data=df_4,drop_first=True)
explicativas = df_4.drop(columns='Me_Gusta')
objetivo = df_4['Me_Gusta']
model = DecisionTreeClassifier()
model.fit(X=explicativas,y=objetivo)
feature_names = df_4.columns.tolist()
a = explicativas.sample()
print(a)
print(model.predict_proba(a))
print(model.score(X=explicativas,y=objetivo))
