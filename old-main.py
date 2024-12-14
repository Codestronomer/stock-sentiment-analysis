import numpy as np
from urllib.request import urlopen, Request
from bs4 import BeautifulSoup
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import pandas as pd
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor, as_completed
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import yfinance as yf
from datetime import datetime, timedelta

finviz_url = 'https://finviz.com/quote.ashx?t='
tickers = ['AMZN', 'AAPL', 'GOOG', 'NVDA']

news_tables = {}
for ticker in tickers:
  url = finviz_url + ticker
  req = Request(url=url, headers={'user-agent': 'my-app'})
  response = urlopen(req)

  html = BeautifulSoup(response, 'html.parser')
  # print(html) 

  news_table = html.find(id='news-table')
  news_tables[ticker] = news_table

# print(news_tables)

# amzn_data = news_tables['AMZN']
# amzn_rows = amzn_data.findAll('tr')

# for index, row in enumerate(amzn_rows):
#   title = row.a.text
#   timestamp = row.td.text
#   print(f'{timestamp}:  {title}')

def parse_date(date_str):
    try:
        pd.to_datetime(date_str).date()
    except ValueError:
        print(date_str)
        return None
     

parsed_data = []
for ticker, news_table in news_tables.items():
    for row in news_table.findAll('tr'):
        title = row.a.text
        date_data = row.td.text.strip().split(' ')
        
        if len(date_data) == 1:
            time = date_data[0]
            date = datetime.now().date()
        else:
            date = parse_date(date_data[0])
            time = date_data[1]
        
        parsed_data.append([ticker, date, time, title])

df = pd.DataFrame(parsed_data, columns=['ticker', 'date', 'time', 'title'])

vader = SentimentIntensityAnalyzer()
df['compound'] = df['title'].apply(lambda x: vader.polarity_scores(x)['compound'])

# Fetch historical stock data
def fetch_stock_data(ticker):
    stock_data = yf.download(ticker, start="2022-01-01", end="2024-01-01")
    stock_data['ticker'] = ticker
    return stock_data

all_stock_data = pd.concat([fetch_stock_data(ticker) for ticker in tickers])
print(all_stock_data.head())

all_stock_data.reset_index(inplace=True)

all_stock_data['Date'] = pd.to_datetime(all_stock_data['Date'])
all_stock_data['date'] = all_stock_data['Date'].dt.date

# Merge sentiment data with stock data
merged_df = pd.merge(df, all_stock_data, how='inner', on=['ticker', 'date'])
merged_df['price_change'] = merged_df['Close'].pct_change() 

# Feature Engineering
merged_df['sentiment_rolling_mean'] = merged_df.groupby('ticker')['compound'].rolling(3).mean().reset_index(0, drop=True)
merged_df['price_change_next_day'] = merged_df.groupby('ticker')['price_change'].shift(-1)

# Prepare data for machine learning
features = ['compound', 'sentiment_rolling_mean']
merged_df.dropna(subset=features + ['price_change_next_day'], inplace=True)
X = merged_df[features]
y = (merged_df['price_change_next_day'] > 0).astype(int)  # Binary classification: 1 if price increases, 0 otherwise

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, train_size=0.7, random_state=42)

# Train a Random Forest Classifier
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))

# Visualization
plt.figure(figsize=(10, 8))
mean_df = df.groupby(['ticker', df['date']]).mean(numeric_only=True).unstack()
mean_df = mean_df.xs('compound', axis="columns").transpose()
mean_df.plot(kind='bar', figsize=(15, 7))
plt.title('Sentiment Analysis Over Time')
plt.xlabel('Date')
plt.ylabel('Average Sentiment Score')
plt.legend(title='Ticker')
plt.show()
# df['date'] = pd.to_datetime(df.date).dt.date

# plt.figure(figsize=(10, 8))

# mean_df = df.groupby(['ticker', 'date']).mean(numeric_only=True)

# mean_df = mean_df.unstack()
# mean_df = mean_df.xs('compound', axis="columns").transpose()
# mean_df.plot(kind='bar')
# plt.show()
# print(mean_df)