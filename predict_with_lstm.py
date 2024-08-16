import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM
from utils.coingecko import get_coingecko_data

# Plit data into training and testing sets
data = get_coingecko_data('eth')
df = pd.DataFrame(data["prices"])
df.columns = ["timestamp", "price"]
df["timestamp"] = pd.to_datetime(df["timestamp"], unit='ms')
df.set_index("timestamp", inplace=True)
# Normalize the price data
scaler = MinMaxScaler(feature_range=(0, 1))
df['price'] = scaler.fit_transform(df['price'].values.reshape(-1, 1))


train_size = int(len(df) * 0.8)
test_size = len(df) - train_size
train_data, test_data = df.iloc[:train_size], df.iloc[train_size:]

# Data Preparation for LSTM
def create_dataset(dataset, look_back=1):
  dataX, dataY = [], []
  for i in range(len(dataset)-look_back-1):
    a = dataset[i:(i+look_back), 0]
    dataX.append(a)
    dataY.append(dataset[i + look_back, 0])
  return np.array(dataX), np.array(dataY)

# Create the dataset
look_back = 30
trainX, trainY = create_dataset(train_data.values, look_back)
testX, testY = create_dataset(test_data.values, look_back)

# Reshape input to be [samples, time steps, features]
trainX = np.reshape(trainX, (trainX.shape[0], trainX.shape[1], 1))
testX = np.reshape(testX, (testX.shape[0], testX.shape[1], 1))

# Create the LSTM model
model = Sequential()
model.add(LSTM(50, input_shape=(look_back, 1)))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')

model.fit(trainX, trainY, epochs=100, batch_size=1, verbose=2)

# Make predictions
trainPredict = model.predict(trainX)
testPredict = model.predict(testX)

# Invert predictions
trainPredict = scaler.inverse_transform(trainPredict)
trainY = scaler.inverse_transform([trainY])
testPredict = scaler.inverse_transform(testPredict)
testY = scaler.inverse_transform([testY])

latest_data = np.array(df.tail(look_back))
latest_data = np.reshape(latest_data, (1, look_back, 1))

# Dự đoán giá trị mới nhất
predicted_price = model.predict(latest_data)

# Chuyển đổi giá trị dự đoán về giá trị gốc
predicted_price = scaler.inverse_transform(predicted_price)

# In ra giá trị dự đoán
print("Giá trị dự đoán cho thời điểm gần nhất:", predicted_price[0][0])