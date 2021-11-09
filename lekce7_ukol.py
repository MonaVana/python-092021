import pandas
import requests
import statsmodels.formula.api as smf
r = requests.get("https://raw.githubusercontent.com/pesikj/progr2-python/master/data/Fish.csv")
with open("Fish.csv", "wb") as f:
  f.write(r.content)
r = requests.get("https://raw.githubusercontent.com/pesikj/progr2-python/master/data/Concrete_Data_Yeh.csv")
with open("Concrete_Data_Yeh.csv", "wb") as f:
  f.write(r.content)

#Kvalita cementu:
df = pandas.read_csv("Concrete_Data_Yeh.csv")
mod = smf.ols(formula="csMPa ~ cement + slag + flyash + water + superplasticizer + coarseaggregate + fineaggregate + age", data=df)
res = mod.fit()
6print(res.summary())
# R-squared: 0.616, model vysvětluje cca 62 % rozptylu hodnot
# záporný koeficient má voda, asi čím více vody, tím méně pevný beton?

#Ryby:
ryby = pandas.read_csv("Fish.csv")
mod = smf.ols(formula="Weight ~ Length2 + Height", data=ryby)
res = mod.fit()
print(res.summary())
# R-squared: 0.844 kvalita modelu pouze na základě délky ryby
# R-squared: 0.875 kvalita modelu po přidání výšky ryby
prumery = ryby.groupby("Species")["Weight"].mean()
ryby["druh_prum_hmotnost"] = ryby["Species"].map(prumery)
mod = smf.ols(formula="Weight ~ Length2 + Height + druh_prum_hmotnost", data=ryby)
res = mod.fit()
print(res.summary())
#R-squared: 0.900 kvalita modelu po zapracování druhu ryby pomocí target encoding
