from alpha_vantage.timeseries import TimeSeries
import pandas as pd
import numpy as np
from sklearn.discriminant_analysis import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from statsmodels.tsa.arima.model import ARIMA
import statsmodels.api as sm
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.style.use('ggplot')
import math, random
from datetime import datetime, timedelta
import yfinance as yf
import re
from sklearn.linear_model import LinearRegression
from textblob import TextBlob
from bs4 import BeautifulSoup
import requests
import nltk
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM
nltk.download('punkt')


# Ignore Warnings
import warnings
warnings.filterwarnings("ignore")
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


#**************** FUNCTIONS TO FETCH DATA ***************************
def get_historical(quote):
    end = datetime.now()
    start = datetime(end.year-2, end.month, end.day)
    data = yf.download(quote, start=start, end=end)
    df = pd.DataFrame(data=data)

    if(df.empty):
        print(f"yfinance data for {quote} is empty. Fetching from Alpha Vantage...")
        ts = TimeSeries(key='N6A6QT6IBFJOPJ70', output_format='pandas')

        try:
            data, meta_data = ts.get_daily_adjusted(symbol=f'NSE:{quote}', outputsize='full')
            #Format df
            #Last 2 yrs rows => 502, in ascending order => ::-1
            data=data.head(503).iloc[::-1].reset_index()

            # Rename columns to match expected outcome
            df = data.rename(columns={
                'date': 'Date',
                '1. open': 'Open',
                '2. high': 'High',
                '3. low': 'Low',
                '4. close': 'Close',
                '5. adjusted close': 'Adj Close',
                '6. volume': 'Volume'
            })
            df.to_csv(f'data/{quote}.csv', index=False)
        except Exception as e:
            print(f"Error fetch data from Alpha Vantage: {e}");
            return pd.DataFrame()
        
    # Ensure date is a column
    if 'Date' not in df.columns:
        df.reset_index(inplace=True)

    if not df.empty and 'Date' in df.columns:
        df.to_csv(f"data/{quote}.csv", index=False)

    return df

def arima_model(train, test, exog_train, exog_test):
    history = [x for x in train]
    history_exog = [x for x in exog_train]
    predictions = list()
    for t in range(len(test)):
        model = sm.tsa.statespace.SARIMAX(history, exog=history_exog, order=(6, 1, 0))
        model_fit = model.fit(disp=False)
        output = model_fit.forecast(steps=1, exog=[exog_test[t]])
        yhat = output[0]
        predictions.append(yhat)
        obs = test[t]
        history.append(obs)
        history_exog.append(exog_test[t])
    return predictions

def ARIMA_ALGO(df, ticker, split_size):
    unique_values = df["Code"].unique()
    df = df.set_index("Code")

    for company in unique_values[:10]:
        data = (df.loc[company, :]).reset_index()
        data['Price'] = data['Close']
        data['Date'] = pd.to_datetime(data['Date'])
        data = data.set_index('Date')
        data['Price'] = data['Price'].astype(float)
        data = data.fillna(data.bfill())
        quantity = data['Price'].values
        sentiment = data['Sentiment'].values

        fig = plt.figure(figsize=(7.2, 4.8), dpi=65)
        plt.plot(quantity)
        plt.savefig(f'results/graphs/{ticker}/Trends.png')
        plt.close(fig)

        size = int(len(quantity) * 0.80)
        train, test = quantity[:split_size], quantity[split_size:]
        exog_train, exog_test = sentiment[:split_size], sentiment[split_size:]

        # Fit the model
        predictions = arima_model(train, test, exog_train, exog_test)

        # Plot the graph
        fig = plt.figure(figsize=(7.2, 4.8), dpi=65)
        plt.plot(test, label='Actual Price')
        plt.plot(predictions, label='Predicted Price')
        plt.legend(loc=4)
        plt.savefig(f'results/graphs/{ticker}/ARIMA.png')
        plt.close(fig)

        print("##############################################################################")
        arima_pred = predictions[-2]
        print("Tomorrow's", ticker, "Closing Price Prediction by ARIMA:", arima_pred)

        # RMSE calculation
        error_arima = math.sqrt(mean_squared_error(test, predictions))
        print("ARIMA RMSE:", error_arima)
        print("##############################################################################")
        return arima_pred, predictions, error_arima


#************* LSTM SECTION *********************
def LSTM_ALGO(df, ticker, split_size):
    # Split data into training set and test set
    dataset_train = df.iloc[0:split_size, :]
    dataset_test = df.iloc[split_size:, :]

    # Prepare training data
    training_set = df.iloc[:, 4:5].values  # Close price
    sentiment_set = df.iloc[:, -1].values  # Assuming the last column is sentiment

    # Feature Scaling
    sc = MinMaxScaler(feature_range=(0, 1))
    training_set_scaled = sc.fit_transform(training_set)
    sentiment_scaled = MinMaxScaler(feature_range=(0, 1)).fit_transform(sentiment_set.reshape(-1, 1))

    # Creating data structure with 7 timesteps and 1 output
    X_train = []  # memory with 7 days from day i
    y_train = []  # day i
    for i in range(7, len(training_set_scaled)):
        X_train.append(training_set_scaled[i-7:i, 0])
        y_train.append(training_set_scaled[i, 0])
    
    X_train, y_train = np.array(X_train), np.array(y_train)

    # Reshape data for LSTM
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))  # (samples, timesteps, features)
    
    # Initialize RNN
    regressor = Sequential()
    
    # Add LSTM layers with Dropout
    regressor.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
    regressor.add(Dropout(0.1))
    regressor.add(LSTM(units=50, return_sequences=True))
    regressor.add(Dropout(0.1))
    regressor.add(LSTM(units=50, return_sequences=True))
    regressor.add(Dropout(0.1))
    regressor.add(LSTM(units=50))
    regressor.add(Dropout(0.1))
    
    # Add output layer
    regressor.add(Dense(units=1))
    
    # Compile the model
    regressor.compile(optimizer='adam', loss='mean_squared_error')
    
    # Train the model
    regressor.fit(X_train, y_train, epochs=25, batch_size=32)
    
    # Prepare testing data
    real_stock_price = dataset_test.iloc[:, 4:5].values
    sentiment_test_set = sentiment_set[split_size:]

    # Combine train and test set to get the entire dataset
    dataset_total = pd.concat((dataset_train['Close'], dataset_test['Close']), axis=0)
    sentiment_total = np.concatenate((sentiment_scaled[:split_size], sentiment_scaled[split_size:]), axis=0)
    testing_set = dataset_total[len(dataset_total) - len(dataset_test) - 7:].values
    testing_sentiment_set = sentiment_total[len(sentiment_total) - len(dataset_test) - 7:]
    
    # Feature scaling
    testing_set = sc.transform(testing_set.reshape(-1, 1))
    testing_sentiment_set = MinMaxScaler(feature_range=(0, 1)).fit_transform(testing_sentiment_set.reshape(-1, 1))

    # Create data structure for testing
    X_test = []
    for i in range(7, len(testing_set)):
        X_test.append(testing_set[i-7:i, 0])
    
    X_test = np.array(X_test)
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
    
    # Testing prediction
    predicted_stock_price = regressor.predict(X_test)
    
    # Getting original prices back from scaled values
    predicted_stock_price = sc.inverse_transform(predicted_stock_price)
    
    # Prepare X_forecast for forecasting
    X_forecast = testing_set[-7:].reshape(1, 7, 1)

    # Forecasting prediction
    forecasted_stock_price = regressor.predict(X_forecast)
    forecasted_stock_price = sc.inverse_transform(forecasted_stock_price)

    # Calculate error for the testing predictions
    error_lstm = math.sqrt(mean_squared_error(real_stock_price, predicted_stock_price))

    return forecasted_stock_price[0][0], predicted_stock_price, error_lstm


#***************** LINEAR REGRESSION SECTION ******************       
def LIN_REG_ALGO(df, ticker, split_size):
    # Number of days to be forecasted in future
    forecast_out = int(7)

    # Price after n days
    df['Close after n days'] = df['Close'].shift(-forecast_out)

    # New df with only relevant data
    df_new = df[['Close', 'Sentiment', 'Close after n days']]

    # Labels of known data, discard last 7 rows
    y = np.array(df_new.iloc[:-forecast_out, -1])

    # All cols of known data except labels, discard last 7 rows
    X = np.array(df_new.iloc[:-forecast_out, 0:-1])

    # Unknown, X to be forecasted
    X_to_be_forecasted = np.array(df_new.iloc[-forecast_out:, 0:-1])

    # Training, testing to plot graphs, check accuracy
    X_train, X_test = X[:split_size], X[split_size:]
    y_train, y_test = y[:split_size], y[split_size:]

    # Feature Scaling === Normalization
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)
    X_to_be_forecasted = sc.transform(X_to_be_forecasted)

    # Training
    clf = LinearRegression(n_jobs=-1)
    clf.fit(X_train, y_train)

    # Testing
    y_test_pred = clf.predict(X_test)

    # Optional Adjustment: Ensure proper scaling adjustment is documented
    # y_test_pred = y_test_pred * (1.04)

    # Reshape for consistency
    y_test = np.array(y_test)  # Ensure y_test is a flat array if required

    # Plotting
    os.makedirs(f'results/graphs/{ticker}', exist_ok=True)  # Ensure path exists
    fig = plt.figure(figsize=(7.2, 4.8), dpi=65)
    plt.plot(y_test, label='Actual Price')
    plt.plot(y_test_pred, label='Predicted Price')
    plt.legend(loc=4)
    plt.savefig(f'results/graphs/{ticker}/LR.png')
    plt.close(fig)

    # Error Calculation
    error_lr = math.sqrt(mean_squared_error(y_test, y_test_pred))

    # Forecasting
    forecast_set = clf.predict(X_to_be_forecasted)
    forecast_set = forecast_set * (1.04)  # Optional adjustment

    # Metrics
    mean = forecast_set.mean()
    lr_pred = forecast_set[0]  # Ensure correct indexing for first prediction

    print()
    print("##############################################################################")
    print(f"Tomorrow's {ticker} Closing Price Prediction by Linear Regression: {lr_pred}")
    print(f"Linear Regression RMSE: {error_lr}")
    print("##############################################################################")

    return df, lr_pred, forecast_set, mean, error_lr, y_test, y_test_pred


#**************** SENTIMENT ANALYSIS **************************
def parse_date(date_str):
    if date_str == "Today":
        return datetime.now().date()
    elif date_str == "Yesterday":
        return (datetime.now() - timedelta(1)).date()
    else:
        return pd.to_datetime(date_str).date()

def retrieving_news_polarity(symbol):
    stock_ticker_map = pd.read_csv('data/Yahoo-Finance-Ticker-Symbols.csv')
    stock_full_form = stock_ticker_map[stock_ticker_map['Ticker'] == symbol]

    finviz_url = f'https://finviz.com/quote.ashx?t={symbol}'
    headers = {'User-Agent': 'Mozilla/5.0'}
    response = requests.get(finviz_url, headers=headers)

    if response.status_code != 200:
        raise Exception(f"Error fetching data from Finviz: {response.status_code}")

    soup = BeautifulSoup(response.content, 'html.parser')
    news_table = soup.find(id='news-table')

    if not news_table:
        raise Exception("Could not find news table on Finviz page.")

    parsed_data = []
    for row in news_table.findAll('tr'):
        title = row.a.text
        date_data = row.td.text.strip().split(' ')
        
        if len(date_data) == 1:
            time = date_data[0]
            date = datetime.now().date()
        else:
            date = parse_date(date_data[0])
            time = date_data[1]
        
        parsed_data.append([symbol, date, time, title])

    df = pd.DataFrame(parsed_data, columns=['ticker', 'date', 'time', 'title'])

    news_list = []
    global_polarity = 0
    news_texts = []
    daily_polarities = {}
    pos = 0
    neg = 0
    neutral = 0

    for index, row in df.iterrows():
        news_text = row['title']
        news_cleaned = row['title']

        news_cleaned = re.sub('&amp;', '&', news_cleaned)
        news_cleaned = re.sub(':', '', news_cleaned)
        news_cleaned = news_cleaned.encode('ascii', 'ignore').decode('ascii')

        blob = TextBlob(news_cleaned)
        polarity = 0
        for sentence in blob.sentences:
            sentence_polarity = sentence.sentiment.polarity
            polarity += sentence_polarity
            global_polarity += sentence_polarity

            if sentence_polarity > 0:
                pos += 1
            elif sentence_polarity < 0:
                neg += 1
            else:
                neutral += 1

        news_list.append([news_cleaned, polarity])
        news_texts.append(news_text)
        
        if row['date'] not in daily_polarities:
            daily_polarities[row['date']] = []
        daily_polarities[row['date']].append(polarity)

    for date, polarities in daily_polarities.items():
        daily_polarities[date] = sum(polarities) / len(polarities)

    if len(news_list) != 0:
        global_polarity = global_polarity / len(news_list)

    print()
    print("##############################################################################")
    print(f"Positive News: {pos}, Negative News: {neg}, Neutral News: {neutral}")
    print("##############################################################################")

    labels = ['Positive', 'Negative', 'Neutral']
    sizes = [pos, neg, neutral]
    explode = (0.1, 0.1, 0.1)
    fig1, ax1 = plt.subplots(figsize=(7.2, 4.8), dpi=65)
    ax1.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%', startangle=90)
    ax1.axis('equal')
    plt.tight_layout()
    plt.savefig(f'results/graphs/{symbol}/SA.png')
    plt.close(fig1)

    news_polarity = "Overall Positive" if global_polarity > 0 else "Overall Negative"

    print()
    print("##############################################################################")
    print(f"News Polarity: {news_polarity}")
    print("##############################################################################")

    return global_polarity, news_texts, news_polarity, pos, neg, neutral, daily_polarities


# Update Dataset with Sentiment Scores
def update_dataset_with_sentiment(df, daily_polarities):
    df['Date'] = pd.to_datetime(df['Date']).dt.date
    df['Sentiment'] = df['Date'].apply(lambda x: daily_polarities.get(x, 0))
    return df


def recommending(df, global_polarity, today_stock, mean):
  
    # Ensure data is numeric
    today_stock['Close'] = pd.to_numeric(today_stock['Close'], errors="coerce");
    mean = float(mean)

    if today_stock.iloc[-1]['Close'] < mean:
        if global_polarity > 0:
            idea="RISE"
            decision="BUY"
            print()
            print("##############################################################################")
            print("According to the ML Predictions and Sentiment Analysis of Tweets, a",idea,"in",quote,"stock is expected => ",decision)
        elif global_polarity <= 0:
            idea="FALL"
            decision="SELL"
            print()
            print("##############################################################################")
            print("According to the ML Predictions and Sentiment Analysis of Tweets, a",idea,"in",quote,"stock is expected => ",decision)
    else:
        idea="FALL"
        decision="SELL"
        print()
        print("##############################################################################")
        print("According to the ML Predictions and Sentiment Analysis of Tweets, a",idea,"in",quote,"stock is expected => ",decision)
    return idea, decision


def plot_predictions(real, predicted, title, filename):
    print("Length of real_stock_price:", len(real))
    print("Length of arima_forecast:", len(predicted))
    real = real.ravel()
    plt.figure(figsize=(10, 6))
    plt.plot(real, color='blue', label='Actual Stock Price')
    plt.plot(predicted, color='red', label='Predicted Stock Price')
    plt.title(title)
    plt.xlabel('Time')
    plt.ylabel('Stock Price')
    plt.legend()
    plt.savefig(filename)
    plt.close()


def initialize_analysis(quote):
    #************** PREPROCESSING ***********************
    try:
        get_historical(quote)
    except:
        print("Quote not found, Enter a valid quote name")
    else:
        df = pd.read_csv(f'data/{quote}.csv')
        print("##############################################################################")
        print(f"Today's {quote} Stock Data: ")

        today_stock = df.iloc[-5:]
        print(today_stock)
        print("##############################################################################")
        df = df.dropna()
        df['Code'] = quote
        columns_order = ['Code'] + [col for col in df.columns if col != 'Code']
        df = df[columns_order]

        global_polarity, news_texts, news_polarity, pos, neg, neutral, daily_polarities = retrieving_news_polarity(quote)
        df = update_dataset_with_sentiment(df, daily_polarities)

        # Split size
        split_size = int(0.8 * len(df))

        arima_pred, arima_forecast, error_arima = ARIMA_ALGO(df, quote, split_size)
        lstm_pred, lstm_forecast, error_lstm = LSTM_ALGO(df, quote, split_size)
        df, lr_pred, forecast_set, mean, error_lr, y_test, y_test_pred = LIN_REG_ALGO(df, quote, split_size)

        idea, decision = recommending(df, global_polarity, today_stock, mean)

        print()
        print("Forecasted Prices for Next 7 days:")
        print(forecast_set)
        today_stock = today_stock.round(2)

        # Visualization
        plot_predictions(df.iloc[split_size:, 4:5].values, arima_forecast, f"{quote} ARIMA Model Prediction", f'results/graphs/{quote}/ARIMA.png')
        plot_predictions(df.iloc[split_size:, 4:5].values, lstm_forecast, f"{quote} LSTM Model Prediction", f'results/graphs/{quote}/LSTM.png')
        plot_predictions(df.iloc[split_size:, 4:5].values, y_test_pred, f"{quote} Linear Regression Prediction", f'results/graphs/{quote}/LR.png')

        # Calculate additional metrics
        real_stock_price = df.iloc[split_size:, 4:5].values.flatten()  # Flatten the array to 1D

        best_length = min(len(real_stock_price), len(y_test_pred));

        # Truncate forecasts to match the length of real_stock_price
        real_stock_price = real_stock_price[:best_length]
        arima_forecast = arima_forecast[:best_length]
        lstm_forecast = lstm_forecast[:best_length]
        y_test_pred = y_test_pred[:best_length]

        print(f"Shape of real_stock_price: {real_stock_price.shape}")
        print(f"Shape of arima_forecast: {len(arima_forecast)}")
        print(f"Shape of lstm_forecast: {len(lstm_forecast)}")
        print(f"Shape of y_test_pred: {len(y_test_pred)}")

        metrics = {
            'ARIMA': {
                'RMSE': error_arima,
                'MAE': mean_absolute_error(real_stock_price, arima_forecast),
                'R2': r2_score(real_stock_price, arima_forecast)
            },
            'LSTM': {
                'RMSE': error_lstm,
                'MAE': mean_absolute_error(real_stock_price, lstm_forecast),
                'R2': r2_score(real_stock_price, lstm_forecast)
            },
            'Linear Regression': {
                'RMSE': error_lr,
                'MAE': mean_absolute_error(real_stock_price, y_test_pred),  # Use y_test_pred for metrics
                'R2': r2_score(real_stock_price, y_test_pred)
            }
        }

        print("Evaluation Metrics:")
        for model, metric in metrics.items():
            print(f"{model} - RMSE: {metric['RMSE']}, MAE: {metric['MAE']}, R2: {metric['R2']}")

        return {
            'quote': quote,
            'arima_pred': round(arima_pred, 2),
            'lstm_pred': round(lstm_pred, 2),
            'lr_pred': round(lr_pred, 2),  # Assuming lr_pred is a single prediction
            'open_s': today_stock['Open'].to_string(index=False),
            'close_s': today_stock['Close'].to_string(index=False),
            'adj_close': today_stock['Adj Close'].to_string(index=False),
            'news_texts': news_texts,
            'news_polarity': news_polarity,
            'idea': idea,
            'decision': decision,
            'high_s': today_stock['High'].to_string(index=False),
            'low_s': today_stock['Low'].to_string(index=False),
            'vol': today_stock['Volume'].to_string(index=False),
            'forecast_set': forecast_set,
            'error_lr': round(error_lr, 2),
            'error_lstm': round(error_lstm, 2),
            'error_arima': round(error_arima, 2),
            'metrics': metrics
        }    

#  Entry point
if __name__ == '__main__':
    import os
    quote = 'META'
    directory = f'results/graphs/{quote}'
    parent_dir = ''
    path = os.path.join(parent_dir, directory)
    if not os.path.exists(path):
        os.makedirs(path)
    result = initialize_analysis(quote)
    print("result", result)