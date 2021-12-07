import pandas
import requests
import yfinance as yf
import numpy as np
from pandas.plotting import lag_plot
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
from math import sqrt
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

#Rozšířené zadání:
csco_df_1 = csco_df.reset_index()
plt.plot(csco_df_1["Date"], csco_df_1["Close"])
plt.title("Cisco stock price over time")
plt.xlabel("time")
plt.ylabel("price")
plt.show()

train_data, test_data = csco_df_1[:int(len(csco_df_1) * 0.7)], csco_df_1[int(len(csco_df_1) * 0.7):]
train_data = train_data["Close"].values
test_data = test_data["Close"].values

history = [x for x in train_data]
model_predictions = []
N_test_observations = len(test_data)

for time_point in range(N_test_observations):
  model = ARIMA(history, order=(4,1,0))
  model_fit = model.fit()
  output = model_fit.forecast()
  yhat = output[0]
  model_predictions.append(yhat)
  true_test_value = test_data[time_point]
  history.append(true_test_value)

MSE_error = sqrt(mean_squared_error(test_data, model_predictions))
print('Testing Mean Squared Error: %.3f' % MSE_error)

test_set_range = csco_df_1[int(len(csco_df_1) * 0.7):].index
plt.plot(test_set_range, model_predictions, color="blue", label="Price Prediction")
plt.plot(test_set_range, test_data, color="red", label="Actual Price")

plt.title("Cisco Prices Prediction")
plt.xlabel("Date")
plt.ylabel("Prices")
plt.legend()
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