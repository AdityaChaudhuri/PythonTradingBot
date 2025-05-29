# Trading Bot Code Explanation

## Table of Contents
1. [App.py Overview](#apppy-overview)
2. [Backtester.py Overview](#backtesterpy-overview)
3. [Detailed Code Explanation](#detailed-code-explanation)

## App.py Overview

The `app.py` file is the main application file that creates the web interface using Streamlit. It handles:
- User authentication
- Trading interface
- Real-time data display
- Backtesting interface
- Trade execution

## Backtester.py Overview

The `backtester.py` file contains the backtesting engine that:
- Fetches historical data
- Calculates technical indicators
- Simulates trading strategies
- Generates performance metrics

## Detailed Code Explanation

### App.py Line by Line

```python
import streamlit as st
import pandas as pd
import pandas_ta as ta
import plotly.graph_objects as go
```
- `streamlit`: Web interface framework
- `pandas`: Data manipulation library
- `pandas_ta`: Technical analysis library
- `plotly`: Interactive plotting library

```python
from datetime import datetime, timezone, timedelta
import time
import os
from dotenv import load_dotenv
```
- Time handling imports
- Environment variable management
- Operating system interface

```python
from oandapyV20 import API
import oandapyV20.endpoints.instruments as instruments
import oandapyV20.endpoints.orders as orders
```
- OANDA API integration for:
  - Market data access
  - Order execution
  - Account management

```python
# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
```
- Configures logging system
- Tracks application events and errors
- Provides debugging information

```python
def rate_limit(seconds=1):
    """Rate limiting decorator"""
```
- Prevents API request overload
- Ensures minimum time between requests
- Protects against API limits

```python
# Initialize session state
if 'authenticated' not in st.session_state:
    st.session_state.authenticated = False
```
- Manages user session data
- Tracks authentication status
- Stores trading parameters

```python
def is_valid_api_key(api_key):
    """Basic validation of API key format"""
```
- Validates API key format
- Basic security check
- Prevents invalid API requests

```python
@handle_api_errors
def get_candles(tf, instrument, client):
    """Fetch candles with rate limiting and error handling"""
```
- Fetches market price data
- Implements rate limiting
- Handles API errors

```python
def calculate_indicators(df):
    """Calculate technical indicators"""
```
- Computes EMAs, RSI, MACD
- Adds Bollinger Bands
- Calculates volume indicators

### Backtester.py Line by Line

```python
class Backtester:
    def __init__(self, client, instrument, timeframe, initial_balance=10000):
```
- Initializes backtesting engine
- Sets up trading parameters
- Configures initial account state

```python
def get_historical_data(self, start_date, end_date=None, progress_bar=None):
    """Fetch historical data from OANDA using pagination"""
```
- Fetches historical price data
- Handles date ranges
- Shows progress updates

```python
# Validate and adjust dates
current_time = datetime.now(timezone.utc)
```
- Ensures valid date ranges
- Handles timezone conversions
- Prevents future date requests

```python
# Initialize empty list for all candles
all_candles = []
current_start = start_date
chunk_size = timedelta(days=3)
```
- Manages data chunking
- Optimizes data fetching
- Prevents memory issues

```python
def calculate_indicators(self, df, progress_bar=None):
    """Calculate technical indicators with progress tracking"""
```
- Computes trading indicators
- Shows calculation progress
- Handles data validation

```python
def check_signals(self, row, prev_row):
    """Check for trading signals"""
```
- Analyzes market conditions
- Identifies trading opportunities
- Applies strategy rules

```python
def run_backtest_simulation(self, df, risk_reward=1.5, progress_bar=None):
    """Run backtest simulation on pre-calculated data"""
```
- Simulates trading strategy
- Tracks performance metrics
- Manages risk parameters

```python
def plot_results(self, results, df):
    """Plot backtest results"""
```
- Creates interactive charts
- Visualizes trade history
- Shows performance metrics

### Key Trading Strategy Components

The trading strategy uses multiple indicators:

1. **Trend Following**
```python
trend_up = (row['EMA_20'] > row['EMA_50'] and
           row['close'] > row['EMA_20'])
```
- Uses EMAs to identify trends
- Confirms price momentum
- Reduces false signals

2. **Momentum**
```python
rsi_bullish = 30 < row['RSI'] < 70
macd_bullish = (row['MACD_hist'] > 0 and 
               prev_row['MACD_hist'] < row['MACD_hist'])
```
- RSI for overbought/oversold
- MACD for momentum confirmation
- Combined signal validation

3. **Volatility**
```python
bb_bullish = row['close'] > row['BB_middle']
atr_stop = entry_price - (row['ATR_14'] * 1.5)
```
- Bollinger Bands for volatility
- ATR for stop loss calculation
- Dynamic position sizing

### Risk Management

```python
risk_amount = self.balance * 0.01  # 1% risk per trade
position_size = risk_amount / (entry_price - stop_loss)
```
- Fixed percentage risk per trade
- Dynamic position sizing
- Stop loss placement

### Performance Metrics

```python
def _calculate_metrics(self, trades_df, equity_curve):
    """Calculate performance metrics"""
```
Calculates:
- Win rate
- Profit factor
- Average win/loss
- Total return
- Maximum drawdown

### Error Handling

```python
@handle_api_errors
def place_order(client, instrument, units, stop_loss, take_profit):
```
- Graceful error handling
- API request retries
- User feedback

### Progress Tracking

```python
if progress_bar is not None:
    progress = min(days_processed / total_days, 0.99)
    progress_bar.progress(progress)
```
- Real-time progress updates
- User interface feedback
- Process monitoring

This documentation provides a comprehensive understanding of how the trading bot works, from data fetching to strategy execution and performance analysis. Each component is designed to work together to create a complete trading system with proper risk management and performance tracking. 