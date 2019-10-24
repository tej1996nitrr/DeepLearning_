import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset_train =pd.read_csv('Google_stock_price_train.csv')
training_set = dataset_train.iloc[:,1:2].values
type(training_set)

from sklearn.preprocessing import  MinMaxScaler
sc = MinMaxScaler(feature_range=(0,1))
training_set_scaled =sc.fit_transform(training_set)

x_train=[]
y_train=[]
#creating a data structure with 60 timestamps(x_train) and 1 output(y_train)
for i in range(60,1258):
    x_train.append(training_set_scaled[i-60:i,0])
    y_train.append(training_set_scaled[i,0])


x_train,y_train =  np.array(x_train),np.array(y_train)
x_train[2]
x_train.shape
#reshaping (no of stock prices, no of timestamps, no of indicators/predictors)
x_train = np.reshape(x_train,(x_train.shape[0],x_train.shape[1],1))
x_train.shape

from keras.models import Sequential
from keras.layers import Dropout,LSTM,Dense

regressor = Sequential()
regressor.add(LSTM(units=50,return_sequences=True,input_shape=(x_train.shape[1],1)))#since we are addding more than one lstm layers, return_seq=true
regressor.add(Dropout(0.2))

regressor.add(LSTM(units=50,return_sequences=True))#since we are addding more than one lstm layers, return_seq=true
regressor.add(Dropout(0.2))

regressor.add(LSTM(units=50,return_sequences=True))#since we are addding more than one lstm layers, return_seq=true
regressor.add(Dropout(0.2))

regressor.add(LSTM(units=50,return_sequences=False))#since we are addding more than one lstm layers, return_seq=true
regressor.add(Dropout(0.2))

regressor.add(Dense(units=1))

regressor.compile(optimizer='adam',loss='mean_squared_error')

regressor.fit(x_train,y_train,epochs=100,batch_size=32)

dataset_test =pd.read_csv('Google_stock_price_test.csv')
real_stock_price = dataset_test.iloc[:,1:2].values #test set

#dataset_train conains till 2016 dataset test contains jan 2017
dataset_total = pd.concat([dataset_train['Open'],dataset_test['Open']],axis=0)
inputs=dataset_total[len(dataset_total)-len(dataset_test)-60:].values #jan 3 2017's index and -60 =>
inputs = inputs.reshape(-1,1)
inputs=sc.transform(inputs) # we are not using fit method because sc was already fitted with training set

x_test=[]
for i in range(60,80):
    x_test.append(inputs[i-60:i,0])
x_test = np.array(x_test)
x_test = np.reshape(x_test,(x_test.shape[0],x_test.shape[1],1))
predicted_stock_price=regressor.predict(x_test)

predicted_stock_price=sc.inverse_transform(predicted_stock_price)
x_test.shape
plt.plot(real_stock_price,color='red',label='Real Google Stock Price')
plt.plot(predicted_stock_price,color='green',label='Predicted Google Stock Price')
plt.title('Google Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Stock Price')
plt.legend()
plt.show()
