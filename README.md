# 📈 Stock Sentiment Analysis and Price Prediction

This project focuses on predicting stock prices using **sentiment analysis** and statistical/machine learning models like **SARIMAX**, **LSTM**, and **Linear Regression**. It combines historical stock price data with sentiment data to forecast stock price trends effectively.

---

## 🚀 Features

- **Sentiment Analysis**: Analyzes text-based data to derive sentiment scores.
- **Stock Price Forecasting**:
  - ARIMA/SARIMAX for time-series prediction.
  - LSTM (Long Short-Term Memory) for deep learning-based forecasting.
  - Linear Regression for quick predictions.
- **Visualization**: Graphs for actual vs predicted stock prices.
- **Comparison**: Compare results of different forecasting models (ARIMA, LSTM, Linear Regression).

---

##  Models Used

  ### 1. Sentiment Analysis
	- Sentiment scores extracted from external data sources (e.g., news headlines, social media).
    - Scores are used as features for price predictions.
	
   ### 2.	Time Series Models
	  - ARIMA: For statistical forecasting based on stock price trends.
	  - LSTM: A deep learning model ideal for sequential data like time series.
	
   ### 3.	Linear Regression
	   - A baseline regression model predicting future stock prices.

## 🛠️ Tech Stack

- **Python 3.11** *(Required)*
- **Pandas**: Data manipulation and analysis.
- **NumPy**: Numerical computations.
- **Scikit-learn**: Machine learning algorithms.
- **TensorFlow/Keras**: Deep learning for LSTM models.
- **Statsmodels**: ARIMA and SARIMAX models.
- **Matplotlib**: Data visualization.
- **NLTK or TextBlob**: Sentiment analysis.
  
---

## ⚙️ Installation

To set up and run the project, follow these steps:

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/stock-sentiment-analysis.git
cd stock-sentiment-analysis
```

### 2.	Create a Virtual Environment
```bash
python -m venv env  
source env/bin/activate      # On Windows: env\Scripts\activate
```
###	3.	Install Dependencies

```bash
pip install -r requirements.txt
```

## 🚀 Usage
  
### Prepare Your Dataset
- Place historical stock prices (CSV) in the data/ folder.

  
### ⚙️ Configuration 


You can modify key parameters in the main script:
- Split Size: Ratio of training/testing data.
- Forecast Days: Number of future days to predict.
- Ticker Symbol: Stock symbol to analyze (e.g., META, AAPL).

### Run the Project
  To execute the analysis and predictions, use the following command:
  ```bash
  python main.py
  ```


### 📊 Output
*	Graphs: Actual vs predicted stock price graphs are saved under the results/graphs/ directory.
*	Forecasts: Future price predictions are displayed in the console.

Sample: 
```bash
Tomorrow's META Closing Price Prediction: 340.25  
Linear Regression RMSE: 5.72  
ARIMA RMSE: 6.11  
LSTM RMSE: 4.89
```

✅ Dependencies
- Python 3.11
- Libraries:
  * pandas
  * numpy
  * matplotlib
  * sklearn
  * tensorflow
  * statsmodels

### 📚 Future Improvements
*	Integrate real-time sentiment analysis using live news APIs.
*	Add support for additional stock market models.
*	Optimize hyperparameters for LSTM and ARIMA.

### 👨‍💻 Contact
- Name: John Rumide (Codestronomer)
- Email: Johnrumide6@gmail.com
- Github: [Codestronomer](https://github.com/codestronomer)

⚖️ License

This project is licensed under the MIT License.

Feel free to adapt it further! Let me know if you need tweaks or additions. 🚀
