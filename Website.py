import streamlit as st
from Stock_Prediction_Test_Learn_LSTM import stock_predict_LSTM
from Stock_Prediction_Test_Learn_LSTM import stock_viz_close
from Stock_Prediction_Test_Learn_LSTM import stock_viz_volume
from Stock_Prediction_LR import linear_predict
from Stock_Prediction_LR import cross_val
from Stock_Prediction_LR import prepare_data

import streamlit.components.v1 as components

st.title("Sam's Stock Predictions.")

st.write("""

Hi everyone!!!
 
Welcome to my first python project. 

This is a website about the stock market...

First and foremost, I am not certified in anyway, shape, or form to give financial advice.
I am no Finance expert by any means, I just thought this would be a fun project.  

I am using a Long Short-Term Memory Neural Network to predict the stock market.  
Anytime the Root-Means-Square-Error is around or less than 5, 
The Neural Network is running optimally. 

I am also using a much more simple algorithm known as linear regression. This is a 2 dimensional way
For the computer to pick up on trends in the stock market. This is a much more simple algorithm that can 
Predict trends in a period of about 5 days. On the x-axis it increments in half day period for a span of five days.
We are looking for an ideal test Score of .95 or higher... 

""")

header = st.sidebar.header("User Input Features")
ticker_close = st.sidebar.selectbox('What stock do you want to see closing prices for?',('NIO', 'AAPL', 'HTZGQ', 'TSLA', 'GOOG'))
ticker_vol = st.sidebar.selectbox('What stock do you want to see volumes for?',('NIO', 'AAPL', 'HTZGQ', 'TSLA', 'GOOG'))
ticker_predict = st.sidebar.selectbox('What stock do you want to see (LSTM) predictions for?',('NIO', 'AAPL', 'HTZGQ', 'TSLA', 'GOOG'))
linear_ticker = st.sidebar.selectbox('What stock do you want to see Linear Regression predictions for?',('NIO', 'AAPL', 'HTZGQ', 'TSLA', 'GOOG'))

stock_viz_close(ticker_close)
stock_viz_volume(ticker_vol)
stock_predict_LSTM(ticker_predict)
linear_predict(linear_ticker)

st.sidebar.header("Future Projects")
st.sidebar.text_input('Let me know what stocks I should add.(Include the ticker please)')
st.sidebar.text_input('What do you want to see in the future?')

st.sidebar.header("Contact Information")
st.sidebar.write("""

----------------------------------------
Work Email - chapmansamuele@gmail.com
----------------------------------------
School Email - samechap@iu.edu
----------------------------------------
Personal Email - chaffylime@gmail.com
----------------------------------------

"""
)
st.sidebar.button('www.linkedin.com/in/sam-chapman-ts-sci')