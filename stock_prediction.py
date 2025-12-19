#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd
import yfinance as yf
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

def load_stock_data(ticker, start_date, end_date):
    """
    Fetches historical stock data from Yahoo Finance.
    """
    print(f"Fetching data for {ticker}...")
    data = yf.download(ticker, start=start_date, end=end_date, progress=False)
    return data

def preprocess_data(data):
    """
    Prepares features (X) and target (y).
    Strategy: Use today's Open, High, Low, Close, Volume to predict 
    TOMORROW'S Close price.
    """
    # Create a new column 'Target' which is the 'Close' price shifted by -1
    # This aligns today's row with tomorrow's price.
    data['Target'] = data['Close'].shift(-1)

    # Drop the last row because it has no 'Target' (we don't know the future of the very last day)
    data = data.dropna()

    # Define Features and Target
    # We use .values to convert to numpy arrays to avoid index alignment issues during splitting
    X = data[['Open', 'High', 'Low', 'Close', 'Volume']].values
    y = data['Target'].values
    
    return X, y

def train_and_evaluate(X, y):
    """
    Trains a Linear Regression model and evaluates it.
    """
    # Split: 80% Training, 20% Testing
    # shuffle=False is often used in time-series, but for a basic regression assignment,
    # random split is acceptable. Here we use standard split.
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize Linear Regression
    model = LinearRegression()

    # Train the model
    print("Training model...")
    model.fit(X_train, y_train)

    # Predictions
    predictions = model.predict(X_test)

    # Metrics
    mse = mean_squared_error(y_test, predictions)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, predictions)

    print("\n--- Model Performance Evaluation ---")
    print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
    print(f"R-squared Score (Accuracy): {r2:.4f}")

    print("\n--- Sample Predictions ---")
    # Show first 5 comparisons
    results = pd.DataFrame({'Actual Next Close': y_test.flatten(), 'Predicted Next Close': predictions.flatten()})
    print(results.head())

    return model

if __name__ == "__main__":
    # Settings
    TICKER = 'AAPL'  # Apple Inc.
    START = '2020-01-01'
    END = '2024-01-01'

    # Execution Pipeline
    df = load_stock_data(TICKER, START, END)
    
    # Check if data loaded correctly
    if not df.empty:
        X, y = preprocess_data(df)
        train_and_evaluate(X, y)
    else:
        print("Error: No data fetched. Check your internet connection or ticker symbol.")


# In[ ]:




