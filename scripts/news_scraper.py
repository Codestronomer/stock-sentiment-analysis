from bs4 import BeautifulSoup
from urllib.request import Request, urlopen

finviz_url = 'https://finviz.com/quote.ashx?t='
tickers = ['AMZN', 'GOOG', 'NVDA', 'META']

def retrieve_news():
  news_tables = {}
  for ticker in tickers:
    url = finviz_url + ticker
    req = Request(url=url, headers={'user-agent': 'my-app'})
    response = urlopen(req)

    html = BeautifulSoup(response, 'html.parser')

    news_table = html.find(id='news-table')
    news_tables[ticker] = news_table

  return news_table