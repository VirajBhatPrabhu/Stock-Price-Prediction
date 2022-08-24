import pandas as pd
import pandas_datareader as pdr
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler

st.markdown("<h1 style='text-align: center;'>STOCK TREND PREDICTION</h1>", unsafe_allow_html=True)

user_input = st.text_input('Please Enter The Stock Ticker', 'AAPL')
start = '2010-01-01'
end = '2022-08-23'
df = pdr.DataReader(user_input, 'yahoo', start=start, end=end)


user_menu= radio = st.radio('Choose An Option', ['Statistics', 'Graph Of the Overall Trend','Closing Value vs Time With 100 Days Moving Average','Closing Value vs Time With 100 & 200 Days Moving Average','Predicted Values vs Original Values'])

if user_menu=='Statistics':
    if st.button('Result'):
        st.markdown("<h3 style='text-align: center;'>Statistics</h3>", unsafe_allow_html=True)
        st.write(df.describe())


if user_menu=='Graph Of the Overall Trend':
    if st.button('Result'):
        st.markdown("<h3 style='text-align: center;'>Graph of Overall Trend</h3>", unsafe_allow_html=True)

        fig = plt.figure(figsize=(15,8))
        plt.plot(df['Close'], 'b')
        plt.xlabel('Time in Days')
        plt.ylabel(f'Close Value of {user_input} Stocks')
        st.pyplot(fig)

if user_menu=='Closing Value vs Time With 100 Days Moving Average':
    if st.button('Result'):
        ma100 = df['Close'].rolling(100).mean()
        st.markdown("<h3 style='text-align: center;'>Closing Value vs Time With 100 Days Moving Average</h3>",unsafe_allow_html=True)
        fig2 = plt.figure(figsize=(15, 8))
        plt.plot(df['Close'], 'b')
        plt.plot(ma100, 'r', label='100 Days Moving Average')
        plt.legend(loc='upper left')
        plt.xlabel('Time')
        plt.ylabel(f'Close Value for {user_input} Stocks')
        st.pyplot(fig2)

if user_menu=='Closing Value vs Time With 100 & 200 Days Moving Average':
    if st.button('Result'):
        ma100 = df['Close'].rolling(100).mean()
        ma200 = df['Close'].rolling(200).mean()
        st.markdown("<h3 style='text-align: center;'>Closing Value vs Time With 100 & 200 Days Moving Average</h3>",
                   unsafe_allow_html=True)

        fig3 = plt.figure(figsize=(15, 8))
        plt.plot(df['Close'], 'b')
        plt.plot(ma100, 'r', label='100 Days Moving Average')
        plt.plot(ma200, 'g', label='200 Days Moving Average')
        plt.legend(loc='upper left')
        plt.xlabel('Time')
        plt.ylabel(f'Close Value of {user_input} Stocks')
        st.pyplot(fig3)

if user_menu=='Predicted Values vs Original Values':
    if st.button('Result'):
        training_data = pd.DataFrame(df.Close[0:int(len(df) * 0.70)])
        test_data = pd.DataFrame(df.Close[int(len(df) * 0.70): int(len(df))])

        scaler = MinMaxScaler(feature_range=(0, 1))
        training_data_array = scaler.fit_transform(training_data)

        model = load_model('Tf_model.h5')

        last100 = training_data.tail(100)
        final_df = last100.append(test_data, ignore_index=True)
        input_data = scaler.fit_transform(final_df)
        X_test = []
        y_test = []
        for i in range(100, input_data.shape[0]):
            X_test.append(input_data[i - 100: i])
            y_test.append(input_data[i, 0])

        X_test, y_test = np.array(X_test), np.array(y_test)

        y_predicted = model.predict(X_test)
        scaler = scaler.scale_

        scalefactor = 1 / scaler[0]
        y_predicted = y_predicted * scalefactor
        y_test = y_test * scalefactor

        st.markdown("<h3 style='text-align: center;'>Predicted Value vs Original Value</h3>",unsafe_allow_html=True)

        fig4 = plt.figure(figsize=(12, 6))
        plt.plot(y_predicted, 'b', label='Oiginal Value')
        plt.plot(y_test, 'g', label='Predicted Value')
        plt.xlabel('Time')
        plt.ylabel(f'Close Value for {user_input}')
        plt.legend(loc='upper left')
        st.pyplot(fig4)



