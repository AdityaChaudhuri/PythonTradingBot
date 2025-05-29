# Trading Bot Code Documentation

## Table of Contents
1. [Project Overview](#project-overview)
2. [File Structure](#file-structure)
3. [Core Components](#core-components)
4. [Detailed Code Documentation](#detailed-code-documentation)
5. [API Integration](#api-integration)
6. [Trading Strategy](#trading-strategy)
7. [Risk Management](#risk-management)
8. [User Interface](#user-interface)

## Project Overview

The Trading Bot is a sophisticated automated trading system built with Python, integrating with the OANDA forex trading platform. It provides real-time market analysis, automated trading execution, and comprehensive backtesting capabilities through an interactive web interface.

## File Structure

```
tradingbot/
├── app.py              # Main application file
├── backtester.py       # Backtesting engine
├── requirements.txt    # Project dependencies
├── .env               # Environment variables
└── trades.json        # Trade history storage
```

## Core Components

### 1. Imports and Setup
```python
import streamlit as st
import pandas as pd
import pandas_ta as ta
import plotly.graph_objects as go
from oandapyV20 import API
```
- `streamlit`: Web interface framework
- `pandas`: Data manipulation and analysis
- `pandas_ta`: Technical analysis tools
- `plotly`: Interactive charting
- `oandapyV20`: OANDA API client

### 2. Session State Management
```python
if 'authenticated' not in st.session_state:
    st.session_state.authenticated = False
```
Manages persistent data across app reruns:
- Authentication status
- Bot running state
- Trading parameters
- Account information

### 3. Rate Limiting
```python
@rate_limit(seconds=1)
def decorator(func):
    # Rate limiting implementation
```
- Prevents API request overload
- Ensures compliance with OANDA limits
- Tracks request timing
- Implements delay when needed

### 4. API Integration

#### Authentication
```python
def is_valid_api_key(api_key):
    """Basic validation of API key format"""
```
- Validates API credentials
- Secures API key storage
- Manages authentication state

#### Market Data
```python
@rate_limit(1)
@handle_api_errors
def get_candles(tf, instrument, client):
    """Fetch candles with rate limiting and error handling"""
```
- Fetches price data
- Implements error handling
- Formats data for analysis

#### Account Management
```python
def get_account_details(client, account_id):
    """Fetch account details with rate limiting and error handling"""
```
- Retrieves account information
- Monitors positions and orders
- Tracks account metrics

### 5. Technical Analysis

#### Indicator Calculation
```python
def calculate_indicators(df):
    """Calculate technical indicators"""
```
Implements multiple technical indicators:
1. Trend Indicators
   - EMA (5, 8, 20, 50 periods)
   - Moving averages

2. Momentum Indicators
   - RSI (14 periods)
   - MACD (12, 26, 9)

3. Volatility Indicators
   - ATR (14 periods)
   - Bollinger Bands (20, 2)

### 6. Trading Strategy

#### Signal Generation
```python
def check_signals(df):
    """Check for trading signals"""
```
Multi-factor analysis:
1. Trend Analysis
   - EMA crossovers
   - Price action

2. Momentum Confirmation
   - RSI conditions
   - MACD momentum

3. Volatility Assessment
   - Bollinger Band position
   - ATR for stop loss

#### Order Management
```python
def place_order(client, instrument, units, stop_loss, take_profit):
    """Place order with error handling"""
```
- Order creation and validation
- Position sizing
- Risk management
- Trade logging

### 7. Risk Management

#### Position Sizing
```python
risk_amount = self.balance * 0.01  # 1% risk per trade
position_size = risk_amount / (entry_price - stop_loss)
```
- Fixed percentage risk
- Dynamic position sizing
- Account balance consideration

#### Trade Limits
```python
def reset_daily_trades():
    """Reset daily trade counter if it's a new day"""
```
- Daily trade limits
- Maximum position size
- Risk per trade limits

### 8. Backtesting Engine

#### Historical Data Management
```python
def get_historical_data(self, start_date, end_date=None, progress_bar=None):
    """Fetch historical data from OANDA using pagination"""
```
- Efficient data fetching
- Progress tracking
- Data validation

#### Performance Analysis
```python
def _calculate_metrics(self, trades_df, equity_curve):
    """Calculate performance metrics"""
```
Calculates key metrics:
- Win rate
- Profit factor
- Average win/loss
- Maximum drawdown
- Total return

### 9. User Interface

#### Dashboard Layout
```python
def main():
    st.title("Trading Bot Dashboard 📈")
```
Main sections:
1. Account Overview
   - Balance and P/L
   - Open positions
   - Margin metrics

2. Trading Interface
   - Price charts
   - Indicator displays
   - Trading controls

3. Backtesting Interface
   - Date range selection
   - Parameter configuration
   - Results visualization

#### Interactive Components
```python
tabs = ["Chart", "Active Trades", "Trade History", "Backtesting"]
```
- Real-time charts
- Trade monitoring
- Performance metrics
- Configuration controls

### 10. Error Handling

#### API Error Management
```python
@handle_api_errors
def wrapper(*args, **kwargs):
    """Error handling wrapper"""
```
- Graceful error recovery
- User notifications
- Error logging
- Rate limit management

#### System Recovery
```python
if st.session_state.error_count >= 5:
    st.session_state.bot_running = False
```
- Error count tracking
- Automatic bot shutdown
- System state management

## API Integration

### OANDA API Endpoints

1. Account Endpoints
```python
accounts.AccountSummary
accounts.AccountInstruments
```
- Account information
- Available instruments
- Trading permissions

2. Instrument Endpoints
```python
instruments.InstrumentsCandles
```
- Price data
- Historical candles
- Instrument details

3. Order Endpoints
```python
orders.OrderCreate
```
- Order placement
- Order modification
- Order cancellation

## Trading Strategy

### Signal Generation Logic

1. Trend Following
```python
trend_up = (row['EMA_20'] > row['EMA_50'] and
           row['close'] > row['EMA_20'])
```
- Moving average alignment
- Price momentum
- Trend confirmation

2. Momentum Analysis
```python
rsi_bullish = 30 < row['RSI'] < 70
macd_bullish = (row['MACD_hist'] > 0 and 
               prev_row['MACD_hist'] < row['MACD_hist'])
```
- RSI conditions
- MACD crossovers
- Momentum confirmation

3. Volatility Assessment
```python
bb_bullish = row['close'] > row['BB_middle']
atr_stop = entry_price - (row['ATR_14'] * 1.5)
```
- Bollinger Band position
- ATR-based stops
- Volatility measurement

## Risk Management

### Position Sizing
```python
risk_amount = balance * 0.01  # 1% risk per trade
position_size = risk_amount / (entry_price - stop_loss)
```
- Fixed percentage risk
- Dynamic sizing
- Account protection

### Trade Management
```python
MAX_DAILY_TRADES = 10
MAX_POSITION_SIZE = 10000
RISK_PERCENTAGE = 1
```
- Daily limits
- Position limits
- Risk controls

## Best Practices

1. **API Usage**
   - Implement rate limiting
   - Handle errors gracefully
   - Validate responses

2. **Risk Management**
   - Fixed percentage risk
   - Position size limits
   - Daily trade limits

3. **Code Organization**
   - Modular structure
   - Clear documentation
   - Error handling

4. **Data Management**
   - Efficient storage
   - Regular backups
   - Data validation

## Troubleshooting

### Common Issues

1. **API Errors**
   - Check credentials
   - Verify rate limits
   - Validate parameters

2. **Performance Issues**
   - Monitor memory usage
   - Optimize data handling
   - Cache when possible

3. **Trading Errors**
   - Verify account balance
   - Check position limits
   - Validate orders

## Maintenance

### Regular Tasks

1. **Daily**
   - Monitor error logs
   - Check trade limits
   - Verify API status

2. **Weekly**
   - Backup trade data
   - Review performance
   - Update parameters

3. **Monthly**
   - Strategy optimization
   - Risk assessment
   - System updates 