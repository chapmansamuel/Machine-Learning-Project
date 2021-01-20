import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn import preprocessing
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from pandas_datareader.nasdaq_trader import get_nasdaq_symbols
import pandas_datareader as web
import matplotlib.pyplot as plt
import streamlit as st

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

def prepare_data(df,forecast_col,forecast_out,test_size):
    label = df[forecast_col].shift(-forecast_out)#creating new column called label with the last 5 rows are nan
    X = np.array(df[[forecast_col]]) #creating the feature array
    X = preprocessing.scale(X) #processing the feature array
    X_lately = X[-forecast_out:] #creating the column i want to use later in the predicting method
    X = X[:-forecast_out] # X that will contain the training and testing
    label.dropna(inplace=True) #dropping na values
    y = np.array(label)  # assigning Y
    Y_lately = y[:-forecast_out] #creating the y prediction value for bayesian statistics 2-d array
    X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=test_size) #cross validation
    response = [X_train, X_test, Y_train, Y_test , X_lately, Y_lately]
    return response

def cross_val(fitment, x_train, y_train):
    crossval = cross_val_score(fitment, x_train, y_train, cv=3)
    return crossval

# def f(x, stock):
#     y = np.sqrt(x) * np.sin(x)
#     poly = np.random.normal(0, 1, len(x))
#     return y + stock * poly
#
# degree = 10
# X = np.linspace(0, 10, 100)
# y = f(X, noise_amount=0.1)
# clf = linear_model.BayesianRidge()
# clf.fit(np.vander(X, degree), y)


def linear_predict(iterations, stock):
    df = web.DataReader(stock, 'stooq')
    df.sort_values('Date', ascending=True, inplace=True)
    data = df.filter(['Close'])

    forecast_col = 'Close'#choosing which column to forecast
    forecast_out = 6 #how far to forecast
    test_size = 0.3 #the size of my test set

    dataset = prepare_data(data,forecast_col,forecast_out,test_size) #calling the method where the cross validation and data preperation is in

    for i in range(len(dataset)):
        X_train = dataset[0]
        X_test = dataset[1]
        Y_train = dataset[2]
        Y_test = dataset[3]
        X_lately = dataset[4]
        Y_lately = dataset[5]

    learner = linear_model.BayesianRidge(iterations, tol=0.001, alpha_1=1e-06, alpha_2=1e-06, lambda_1=1e-06, lambda_2=1e-06, alpha_init=None, lambda_init=None, compute_score=False, fit_intercept=True, normalize=False, copy_X=True, verbose=False) #initializing linear regression model
    learner.fit(X_train,Y_train)#training the linear regression model
    score=learner.score(X_test,Y_test)#testing the linear regression model
    crossvalidated = cross_val(learner, X_train, Y_train)
    forecast = learner.predict(X_lately) #set that will contain the forecasted data

    response = {}  # creating json object
    response['Test Score'] = score
    response['Forecast Set'] = forecast
    response['Cross Validation'] = crossvalidated

    st.title('{} Linear Regression Prediction.'.format(stock.upper()))
    st.line_chart(forecast)
    st.title("Model Metrics for Linear Regression.")
    st.subheader("R^2 Value.")
    st.subheader(score)
    st.subheader("Cross Validation Matrix.")
    st.subheader(crossvalidated)

#Visualize the data
# plt.figure(figsize=(16,8))
# plt.title('Linear Regression')
# plt.xlabel('Date', fontsize=18)
# plt.ylabel('Close Price USD ($)', fontsize=18)
# plt.plot(score)
# plt.plot(forecast)
# plt.legend(['Train', 'Validation', 'Predictions'], loc='lower right')
# plt.show()

# Any results you write to the current directory are saved as output.
