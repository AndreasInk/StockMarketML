import pandas as pd
import yfinance as yf

df = yf.download("MSFT", start="2021-01-01", end="2021-05-30", interval="1h")
df.to_csv('/Users/andreas/Desktop/StockMarket/data/MSFT2021.csv')