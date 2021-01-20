import math
from pandas_datareader.nasdaq_trader import get_nasdaq_symbols
import pandas_datareader as web
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
from sklearn.preprocessing import MinMaxScaler
from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt
import streamlit as st

def stock_predict_LSTM(stock, time_steps=3):
    # getting the data
    df = web.DataReader(stock, 'stooq')
    df.sort_values('Date', ascending=True, inplace=True)

    # dataset
    # data has dates and prices
    # dataset has only closing prices

    data = df.filter(['Close'])
    dataset = data.values
    training_data_len = math.ceil(len(dataset) * .9)

    # scaled data
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(dataset)

    # training dataset
    training_data = scaled_data[0:training_data_len, :]
    x_train = []
    y_train = []
    for i in range(time_steps, len(training_data)):
        x_train.append(training_data[i - time_steps:i])
        y_train.append(training_data[i, 0])
    x_train, y_train = np.array(x_train), np.array(y_train)
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

    # print(x_train.shape)
    # creating test datset / x_test and y_test

    test_data = scaled_data[training_data_len - time_steps:, :]
    x_test = []
    y_test = dataset[training_data_len:, :]

    for i in range(time_steps, len(test_data)):
        x_test.append(test_data[i - time_steps:i, 0])
    # convert into numpy array
    x_test = np.array(x_test)
    # reshaping the data into 3 dimensions
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

    #model
    model = Sequential()
    # 1st layer with Dropout regularisation
    # * units = add 100 neurons is the dimensionality of the output space
    # * return_sequences = True to stack LSTM layers so the next LSTM layer has a three-dimensional sequence input
    # * input_shape => Shape of the training dataset
    model.add(LSTM(units=100, return_sequences=True, input_shape=(x_train.shape[1], 1)))
    # 20% of the layers will be dropped
    model.add(Dropout(0.2))
    # 2nd LSTM layer
    # * units = add 50 neurons is the dimensionality of the output space
    # * return_sequences = True to stack LSTM layers so the next LSTM layer has a three-dimensional sequence input
    model.add(LSTM(units=50, return_sequences=True))
    # 20% of the layers will be dropped
    model.add(Dropout(0.2))
    # 3rd LSTM layer
    # * units = add 50 neurons is the dimensionality of the output space
    # * return_sequences = True to stack LSTM layers so the next LSTM layer has a three-dimensional sequence input
    model.add(LSTM(units=50, return_sequences=True))
    # 50% of the layers will be dropped
    model.add(Dropout(0.5))
    # 4th LSTM layer
    # * units = add 50 neurons is the dimensionality of the output space
    model.add(LSTM(units=50))
    # 50% of the layers will be dropped
    model.add(Dropout(0.5))
    # Dense layer that specifies an output of one unit
    model.add(Dense(units=1))

    # compiling model
    model.compile(optimizer='adam', loss='mean_squared_error')

    # training the model
    model.fit(x_train, y_train, batch_size=1, epochs=1)

    # get predicted values
    predictions = model.predict(x_test)
    predictions = scaler.inverse_transform(predictions)

    # RMSE
    rmse = np.sqrt(np.mean(((predictions- y_test)**2)))
    print("Root Means Squared Error:", rmse)

    # Plot the data
    train = data[:training_data_len]
    valid = data[training_data_len:]
    print(predictions.shape, valid.shape)

    valid['Predictions'] = predictions[:, 0]
    st.title('{} Long-Short-Term-Memory.'.format(stock.upper()))
    st.line_chart(valid[['Close','Predictions']])
    st.title("Model Metrics for Long-Short-Term-Memory.")
    st.subheader("Root Means Squared Error")
    st.subheader(rmse)

    # Visualize the data
    # plt.figure(figsize=(16, 8))
    # plt.title('Model')
    # plt.xlabel('Date', fontsize=18)
    # plt.ylabel('Close Price USD ($)', fontsize=18)
    # plt.plot(train['Close'])
    # plt.plot(valid[['Close', 'Predictions']])
    # plt.legend(['Train', 'Validation', 'Predictions'], loc='lower right')
    # plt.show()

def stock_viz_close(stock):
    df = web.DataReader(stock, 'stooq')
    df.sort_values('Date', ascending=True, inplace=True)
    data = df.filter(['Close'])
    st.title('{} closing price.'.format(stock.upper()))
    st.line_chart(data)

def stock_viz_volume(stock):
    df = web.DataReader(stock, 'stooq')
    df.sort_values('Date', ascending=True, inplace=True)
    data = df.filter(['Volume'])
    st.title('{} volume.'.format(stock.upper()))
    st.line_chart(data)
