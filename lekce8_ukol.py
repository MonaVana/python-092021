import pandas
import requests
import yfinance as yf
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.tsa.seasonal import seasonal_decompose
import matplotlib.pyplot as plt
from statsmodels.tsa.holtwinters import ExponentialSmoothing

r = requests.get("https://raw.githubusercontent.com/pesikj/progr2-python/master/data/MLTollsStackOverflow.csv")
with open("MLTollsStackOverflow.csv", "wb") as f:
  f.write(r.content)

#Cena akcie:
csco = yf.Ticker("CSCO")
csco_df = csco.history(period="5y")
csco_close = csco_df["Close"]
print(csco_close.autocorr(lag=7))
plot_acf(csco_close)
plt.show()
model = AutoReg(csco_close, lags=5, trend="t", seasonal=True, period=12)
model_fit = model.fit()
predictions = model_fit.predict(start=csco_df.shape[0], end=csco_df.shape[0] + 4)
csco_forecast = pandas.DataFrame(predictions, columns=["Prediction"])
csco_df = csco.history(period="60d")
csco_with_prediction = pandas.concat([csco_df, csco_forecast])
csco_with_prediction[["Close", "Prediction"]].plot()
plt.show()

#Otázky na Python:
df = pandas.read_csv("MLTollsStackOverflow.csv")
df = df.set_index("month")
decompose = seasonal_decompose(df["python"], model="multiplicative", period=12)
decompose.plot()
plt.show()
mod = ExponentialSmoothing(df["python"], seasonal_periods=12, trend="mul", seasonal="mul", use_boxcox=True, initialization_method="estimated")
res = mod.fit()
df["python"].plot()
plt.show()