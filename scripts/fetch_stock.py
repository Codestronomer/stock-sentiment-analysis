import yfinance as yf

# Fetch historical stock data
def fetch_stock_data(ticker):
    stock_data = yf.download(ticker, start="2022-01-01", end="2024-01-01")
    stock_data['ticker'] = ticker
    return stock_data