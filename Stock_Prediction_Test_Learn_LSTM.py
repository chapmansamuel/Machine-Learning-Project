import math
from pandas_datareader.nasdaq_trader import get_nasdaq_symbols
import pandas_datareader as web
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt
import streamlit as st

def stock_predict_LSTM(stockname):
    df = web.DataReader(stockname, 'stooq')
    df.sort_values('Date', ascending=True, inplace=True)


    #dataset
    data = df.filter(['Close'])
    dataset = data.values
    training_data_len = math.ceil(len(dataset)*.9)

    #scaled data
    scaler = MinMaxScaler(feature_range =(0,1))
    scaled_data = scaler.fit_transform(dataset)

    #training dataset
    training_data = scaled_data[0:training_data_len, :]
    x_train = []
    y_train = []
    for i in range(60, len(training_data)):
        x_train.append(training_data[i-60:i,0])
        y_train.append(training_data[i,0])
    x_train, y_train = np.array(x_train), np.array(y_train)
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
    print(x_train.shape)

    #model
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
    model.add(LSTM(50, return_sequences=False))
    input_layer = Dense(64, input_shape=(60,))
    model.add(input_layer)
    hidden_layer = Dense(128, activation='relu')
    model.add(hidden_layer)
    output_layer = Dense(32)
    model.add(output_layer)
    model.add(Dense(25))
    model.add(Dense(1))

    # compiling model
    model.compile(optimizer='adam', loss='mean_squared_error')

    # training the model
    model.fit(x_train, y_train, batch_size=1, epochs=1)

    test_data = scaled_data[training_data_len - 60:, :]
    x_test = []
    y_test = dataset[training_data_len:, :]
    for i in range(60, len(test_data)):
        x_test.append(test_data[i-60:i, 0])
    x_test = np.array(x_test)
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

    # predicting the values
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
    st.title('{} Long-Short-Term-Memory.'.format(stockname.upper()))
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

def stock_viz_close(stockname):
    df = web.DataReader(stockname, 'stooq')
    df.sort_values('Date', ascending=True, inplace=True)
    data = df.filter(['Close'])
    st.title('{} closing price.'.format(stockname.upper()))
    st.line_chart(data)

def stock_viz_volume(stockname):
    df = web.DataReader(stockname, 'stooq')
    df.sort_values('Date', ascending=True, inplace=True)
    data = df.filter(['Volume'])
    st.title('{} volume.'.format(stockname.upper()))
    st.line_chart(data)