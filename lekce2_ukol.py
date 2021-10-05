import pandas
import requests
import matplotlib.pyplot as plt
import numpy
import seaborn as sns

with requests.get("https://raw.githubusercontent.com/pesikj/progr2-python/master/data/1976-2020-president.csv") as r:
  open("soubory/1976-2020-president.csv", 'w', encoding="utf-8").write(r.text)

with requests.get("https://raw.githubusercontent.com/pesikj/progr2-python/master/data/air_polution_ukol.csv") as r:
  open("soubory/air_polution_ukol.csv", 'w', encoding="utf-8").write(r.text)

#ukol_volby:
president_elections = pandas.read_csv("soubory/1976-2020-president.csv")
president_elections["Rank"] = president_elections.groupby(["year", "state"])["candidatevotes"].rank(ascending=False)
president_elections_winners = president_elections[president_elections["Rank"] == 1]
president_elections_winners["winner_next_year"] = president_elections_winners["party_simplified"].shift(-51)
president_elections_winners["comparison"] = numpy.where(president_elections_winners["winner_next_year"] != president_elections_winners["party_simplified"], 1,0)
president_elections_winners = president_elections_winners.dropna(subset=["winner_next_year"])
president_elections_winners = president_elections_winners.groupby("state")["comparison"].sum().sort_values()
print(president_elections_winners.to_string())

#ukol_castice:
castice = pandas.read_csv("soubory/air_polution_ukol.csv")
castice["date"] = pandas.to_datetime(castice["date"])
castice["month"] = castice["date"].dt.month
castice["year"] = castice["date"].dt.year
castice_pivot = pandas.pivot_table(castice, index="month", columns="year", values="pm25", aggfunc=numpy.mean)
print(castice_pivot)

#doplnek:
sns.heatmap(castice_pivot, annot=True, fmt=".1f")
plt.show()
castice["day"] = castice["date"].dt.dayofweek
castice["day_of_the_week"] = pandas.cut(castice["day"], bins=[-1, 4, 6])
castice_pivot_2 = pandas.pivot_table(castice, index="day_of_the_week", columns="year", values="pm25", aggfunc=numpy.mean)
print(castice_pivot_2)