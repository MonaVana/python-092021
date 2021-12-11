import pandas
import requests
import statistics
from scipy.stats import gmean
import seaborn as sns
import matplotlib.pyplot as plt

r = requests.get("https://raw.githubusercontent.com/pesikj/progr2-python/master/data/crypto_prices.csv")
open("soubory/crypto_prices.csv", "wb").write(r.content)

df = pandas.read_csv("soubory/crypto_prices.csv")

#kryptoměny:
df["PctChange"] = df.groupby("Symbol")["Close"].pct_change()
correl = df.pivot_table(index="Date", columns="Symbol", values="PctChange").corr()
#print(correl)
#sns.heatmap(correl, vmin=-1, vmax=1, annot=True)
sns.jointplot("BTC", "WBTC", correl, kind='scatter', color='seagreen')
sns.jointplot("USDT", "XRP", correl, kind='scatter', color='seagreen')
plt.show()

#tempo růstu:
xmr = df[(df["Symbol"] == "XMR")]
xmr["PctChange"] = xmr.groupby("Symbol")["Close"].pct_change() + 1
xmr.dropna(inplace=True)
x = xmr.groupby("Symbol")["PctChange"].apply(gmean) - 1
# print(x)

#rozšířené zadání:
df["PctChange"] = df.groupby("Symbol")["Close"].pct_change() + 1
df.dropna(inplace=True)
x = df.groupby("Symbol")["PctChange"].apply(gmean) - 1
# print(x)
