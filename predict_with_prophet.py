import pandas as pd
from prophet import Prophet
from utils.coingecko import get_coingecko_data

def prepare_data(data):
    df = pd.DataFrame(data["prices"])
    df.columns = ["ds", "y"]
    df["ds"] = pd.to_datetime(df["ds"], unit='ms')
    df = df[:-1]
    print(df.tail(5))
    return df

def predict_with_prophet(df):
    model = Prophet()
    model.fit(df)
    future = model.make_future_dataframe(periods=1)
    forecast = model.predict(future)
    return forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].iloc[-1]

token = input("Please enter your token (BTC, ETH, BNB, SOL, ARB): ")

data = get_coingecko_data(token)
df = prepare_data(data)
prediction = predict_with_prophet(df)
print(prediction)