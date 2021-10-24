import pandas
import requests
import numpy

r = requests.get("https://raw.githubusercontent.com/lutydlitatova/czechitas-datasets/main/datasets/lexikon-zvirat.csv")
open("soubory/lexikon-zvirat.csv", "wb").write(r.content)


lexikon = pandas.read_csv("soubory/lexikon-zvirat.csv", sep=";").dropna(how="all", axis="columns")
lexikon = lexikon.dropna()
lexikon = lexikon.set_index("id")

#lexikon_1:

def check_url(radek):
  for radek in lexikon.itertuples():
    if isinstance(radek.image_src, str):
      if radek.image_src.startswith("https://zoopraha.cz/images/"):
       if radek.image_src.endswith(".jpg".lower()):
         print({radek.image_src})
       else:
         return radek.title

print(check_url(lexikon))

#lexikon_2:
def popisek(radek):
  titulek = f"{radek.title} preferuje následující typ stravy:{radek.food}. Konkrétně ocení, když mu do misky přistanou převážně {radek.food_note}. Jak to zvíře poznáme: {radek.description}."
  return titulek

lexikon["popisek"] = lexikon.apply(popisek, axis=1)
print(lexikon["popisek"])




