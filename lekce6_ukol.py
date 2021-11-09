import pandas
import requests
from scipy.stats import mannwhitneyu
with requests.get("https://raw.githubusercontent.com/pesikj/progr2-python/master/data/psenice.csv") as r:
  open("psenice.csv", 'w', encoding="utf-8").write(r.text)
with requests.get("https://raw.githubusercontent.com/pesikj/progr2-python/master/data/air_polution_ukol.csv") as r:
    open("air_polution_ukol.csv", 'w', encoding="utf-8").write(r.text)

#psenice:
#H0 - délky zrn pšenice jsou stejné
#H1 - délky zrn pšenice jsou různé

df = pandas.read_csv("psenice.csv")
result = mannwhitneyu(df["Rosa"], df["Canadian"], alternative="two-sided")
print(result)

#MannwhitneyuResult(statistic=4884.5, pvalue=3.522437521029982e-24), pvalue je menší než hladina významnosti 5 %, tedy zamítáme H0

#jemné částice 2:
data = pandas.read_csv("air_polution_ukol.csv")
data["date"] = pandas.to_datetime(data["date"]).dt.strftime("%Y/%m")
#HO průměrné množství částic je v obou měsících stejné
#H1 průměrné množství částic se liší v obou měsících
x = data[data["date"] == "2019/01"]["pm25"]
y = data[data["date"] == "2020/01"]["pm25"]
print(mannwhitneyu(x, y))
#MannwhitneyuResult(statistic=301.0, pvalue=0.011721695410358317), pvalue je menší než hladina významnosti 5 %, tedy zamítáme H0