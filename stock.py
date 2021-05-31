import sklearn
import yfinance as yf
import streamlit as st
import pandas as pd
from pathlib import Path
import pickle
from typing import List
from pydantic import BaseModel
import numpy as np
from sklearn import preprocessing
from datetime import datetime
from sklearn.model_selection import cross_val_score
from dateutil.parser import parse

class MultipleInputs(BaseModel):
    Open: List[float]
    High: List[float]
    Volume: List[float]
    Date: List[float]

def load_regression_model():
    import_dir = Path("models/reg_model.sav")
    model = pickle.load(open(import_dir, "rb"))
    return model

def multi_pred(item: MultipleInputs):
    # reshape inputs
    model_input = []
    for o, h, v in zip(item.Open, item.High, item.Volume):
        model_input.append([o, h, v])
    reg_model = load_regression_model()

    reg_pred = reg_model.predict(model_input)
    
    return {
        "regression_predictions": [float(i) for i in list(reg_pred)]
    }

def multi_pred_future(item: MultipleInputs, days: int):
    model_input = []
    columns = ["Date", "Open", "High","Volume"]
    for d, o, h, v in zip(item.Date, item.Open, item.High, item.Volume):
        time = pd.to_datetime(d, infer_datetime_format=True, utc=True)  
        ##time = datetime.strptime(d, '%Y-%m-%d %H:%M:%S-%f')
        epoch_time = pd.to_datetime(datetime(1970, 1, 1), infer_datetime_format=True, utc=True)
        epoch_time2 = (time - epoch_time).total_seconds()
        print(epoch_time2)
        model_input.append([epoch_time2, o, h, v])
    ##X = np.array(model_input.drop(['prediction'], 1))
    X = model_input
    Y = df['Low']
    #Performing the Regression on the training data
    forecast_time = int(days)
    X_prediction = X[-forecast_time:]
    ratio = 0.9
    split_row = round(ratio * df.shape[0])
    train_df = df.iloc[:split_row]
    test_df = df.iloc[split_row:]
    X_train = model_input
    Y_train = df['Low']
    Y_test = test_df['Low'] 
    clf = load_regression_model()
    clf.fit(X_train, Y_train)
    prediction = (clf.predict(X_prediction))

    last_row = df.tail(1)
    print(last_row['Close'])

    #Sending the SMS if the predicted price of the stock is at least 1 greater than the previous closing price
    if (float(prediction[4]) > (float(last_row['Close'])) + 1):
        print('hello')
    return {
        "regression_predictions": [float(i) for i in list(prediction)]
    }
##df = yf.download("MSFT", start="2018-11-01", end="2020-10-18", interval="1d")
##df.to_csv('/Users/andreas/Desktop/StockMarket/data/MSFT.csv')
df = pd.read_csv('/Users/andreas/Desktop/StockMarket/data/MSFT2021.csv')
st.line_chart(df['Low'])
##df = yf.download("MSFT", start="2021-05-25", end="2021-05-30", interval="1h")
##st.line_chart(df['High'])
preds = multi_pred(df)
pred = multi_pred_future(df, 5)
st.line_chart(preds)
st.header(pred['regression_predictions'][0])