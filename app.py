import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pandas_datareader as data
import yfinance as yf
import streamlit as st
from keras.models import load_model

st.title('Uber pickups in NYC')

user_input = st.text_input('Enter Stock Ticker','AAPL')
df = yf.download(user_input,start='2017-10-02',end ='2023-10-02')

st.write(df.describe())