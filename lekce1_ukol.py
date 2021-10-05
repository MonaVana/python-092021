import pandas
import requests
import matplotlib.pyplot as plt
import numpy
import seaborn as sns

r = requests.get("https://raw.githubusercontent.com/pesikj/progr2-python/master/data/london_merged.csv")
open("soubory/london_merged.csv", 'wb').write(r.content)

r = requests.get("https://raw.githubusercontent.com/pesikj/progr2-python/master/data/titanic.csv")
open("soubory/titanic.csv", 'wb').write(r.content)

#úkol_1
df_titanic = pandas.read_csv("soubory/titanic.csv")
df_titanic_pivot = pandas.pivot_table(df_titanic, index="Pclass", columns="Sex", aggfunc=numpy.sum, values="Survived")
print(df_titanic_pivot)
#rozšířený úkol
df_titanic_1class = df_titanic[df_titanic["Pclass"] == 1]
df_titanic_1class["AgeGroup"] = pandas.cut(df_titanic_1class["Age"], bins=[12, 19, 65])
df_titanic_1class_pivot = pandas.pivot_table(df_titanic_1class, index="AgeGroup", columns="Sex", aggfunc=numpy.sum, values="Survived")
print(df_titanic_1class_pivot)

#úkol_2
df_kola = pandas.read_csv("soubory/london_merged.csv")
df_kola["timestamp"] = pandas.to_datetime(df_kola["timestamp"])
df_kola["year"] = df_kola["timestamp"].dt.year
df_kola_pivot = pandas.pivot_table(df_kola, index="year", columns="weather_code", aggfunc=len, values="cnt", margins=True)
print(df_kola_pivot)
#rozšířený úkol
df_kola_pivot_pct = df_kola_pivot.div(df_kola_pivot.iloc[:,-1], axis=0)
print(df_kola_pivot_pct)
