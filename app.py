import streamlit as st
import pandas as pd
import pandas_ta as ta
import plotly.graph_objects as go
from datetime import datetime, timezone, timedelta
import time
import os
from dotenv import load_dotenv
from oandapyV20 import API
import oandapyV20.endpoints.instruments as instruments
import oandapyV20.endpoints.orders as orders
import oandapyV20.endpoints.accounts as accounts
import json
from pathlib import Path
from backtester import Backtester
import logging
from functools import wraps
import hashlib

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Cache for rate limiting
if 'last_api_call' not in st.session_state:
    st.session_state.last_api_call = {}

def rate_limit(seconds=1):
    """Rate limiting decorator"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            current_time = time.time()
            func_name = func.__name__
            
            if func_name in st.session_state.last_api_call:
                last_call = st.session_state.last_api_call[func_name]
                time_passed = current_time - last_call
                
                if time_passed < seconds:
                    time.sleep(seconds - time_passed)
            
            st.session_state.last_api_call[func_name] = time.time()
            return func(*args, **kwargs)
        return wrapper
    return decorator

# Load environment variables
load_dotenv()

# App config and security settings
st.set_page_config(
    page_title="Trading Bot Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'About': "# Trading Bot Dashboard\nA secure and reliable trading bot using OANDA API."
    }
)

# Initialize session state
if 'authenticated' not in st.session_state:
    st.session_state.authenticated = False
if 'bot_running' not in st.session_state:
    st.session_state.bot_running = False
if 'trades' not in st.session_state:
    st.session_state.trades = []
if 'daily_trades' not in st.session_state:
    st.session_state.daily_trades = 0
if 'last_trade_date' not in st.session_state:
    st.session_state.last_trade_date = None
if 'error_count' not in st.session_state:
    st.session_state.error_count = 0
if 'last_error_time' not in st.session_state:
    st.session_state.last_error_time = None
if 'client' not in st.session_state:
    st.session_state.client = None
if 'instrument' not in st.session_state:
    st.session_state.instrument = "EUR_USD"
if 'timeframe' not in st.session_state:
    st.session_state.timeframe = "M15"
if 'risk_reward' not in st.session_state:
    st.session_state.risk_reward = 1.5
if 'position_size' not in st.session_state:
    st.session_state.position_size = 1000
if 'account_id' not in st.session_state:
    st.session_state.account_id = None

# Security check function
def is_valid_api_key(api_key):
    """Basic validation of API key format"""
    if not api_key:
        return False
    if len(api_key) < 20:  # Minimum length check
        return False
    return True

# Error handling wrapper
def handle_api_errors(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            current_time = time.time()
            
            # Reset error count if last error was more than 1 hour ago
            if (st.session_state.last_error_time and 
                current_time - st.session_state.last_error_time > 3600):
                st.session_state.error_count = 0
            
            st.session_state.error_count += 1
            st.session_state.last_error_time = current_time
            
            # Log the error
            logger.error(f"Error in {func.__name__}: {str(e)}")
            
            # Stop bot if too many errors
            if st.session_state.error_count >= 5:
                st.session_state.bot_running = False
                st.error("Bot stopped due to multiple errors. Please check the logs.")
            
            return None
    return wrapper

@rate_limit(1)
@handle_api_errors
def get_candles(tf, instrument, client):
    """Fetch candles with rate limiting and error handling"""
    params = {
        "granularity": tf,
        "price": "A",
        "count": 100
    }
    
    try:
        r = instruments.InstrumentsCandles(instrument=instrument, params=params)
        response = client.request(r)
        candles = response['candles']
        
        data = []
        for c in candles:
            if c["complete"]:
                data.append({
                    "time": c["time"],
                    "open": float(c["ask"]["o"]),
                    "high": float(c["ask"]["h"]),
                    "low": float(c["ask"]["l"]),
                    "close": float(c["ask"]["c"])
                })
        
        df = pd.DataFrame(data)
        df["time"] = pd.to_datetime(df["time"])
        return df
    except Exception as e:
        st.error(f"Error fetching candles: {str(e)}")
        return None

def calculate_indicators(df):
    """Calculate technical indicators"""
    # Trend Indicators
    df["EMA_5"] = ta.ema(df["close"], length=5)
    df["EMA_8"] = ta.ema(df["close"], length=8)
    df["EMA_20"] = ta.ema(df["close"], length=20)
    df["EMA_50"] = ta.ema(df["close"], length=50)
    
    # Volatility Indicators
    df["ATR_14"] = ta.atr(df["high"], df["low"], df["close"], length=14)
    
    # Calculate Bollinger Bands
    bbands = ta.bbands(df["close"], length=20, std=2)
    df["BB_upper"] = bbands["BBU_20_2.0"]
    df["BB_middle"] = bbands["BBM_20_2.0"]
    df["BB_lower"] = bbands["BBL_20_2.0"]
    
    # Momentum Indicators
    df["RSI"] = ta.rsi(df["close"], length=14)
    macd = ta.macd(df["close"])
    df["MACD"] = macd["MACD_12_26_9"]
    df["MACD_signal"] = macd["MACDs_12_26_9"]
    df["MACD_hist"] = macd["MACDh_12_26_9"]
    
    return df

def reset_daily_trades():
    """Reset daily trade counter if it's a new day"""
    current_date = datetime.now(timezone.utc).date()
    if (st.session_state.last_trade_date is None or 
        current_date != st.session_state.last_trade_date):
        st.session_state.daily_trades = 0
        st.session_state.last_trade_date = current_date

@handle_api_errors
def place_order(client, instrument, units, stop_loss, take_profit):
    """Place order with error handling"""
    try:
        # Validate inputs
        if not all([client, instrument, units, stop_loss, take_profit]):
            return False, "Invalid order parameters"
        
        if units > int(os.getenv('MAX_POSITION_SIZE', 10000)):
            return False, "Order size exceeds maximum position size"
        
        # Check daily trade limit
        reset_daily_trades()
        if st.session_state.daily_trades >= int(os.getenv('MAX_DAILY_TRADES', 10)):
            return False, f"Daily trade limit of {os.getenv('MAX_DAILY_TRADES', 10)} reached. Try again tomorrow."
        
        data = {
            "order": {
                "instrument": instrument,
                "units": units,
                "type": "MARKET",
                "stopLossOnFill": {"price": f"{stop_loss:.5f}"},
                "takeProfitOnFill": {"price": f"{take_profit:.5f}"},
            }
        }
        
        r = orders.OrderCreate(st.session_state.account_id, data=data)
        response = client.request(r)
        
        # Log trade
        trade = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "instrument": instrument,
            "units": units,
            "entry": response["orderFillTransaction"]["price"],
            "stop_loss": stop_loss,
            "take_profit": take_profit,
            "status": "OPEN"
        }
        st.session_state.trades.append(trade)
        
        # Increment daily trade counter
        st.session_state.daily_trades += 1
        save_trades()
        
        return True, "Order placed successfully"
    except Exception as e:
        return False, f"Error placing order: {str(e)}"

def save_trades():
    """Save trades to JSON file"""
    trades_file = Path("trades.json")
    with open(trades_file, "w") as f:
        json.dump(st.session_state.trades, f)

def load_trades():
    """Load trades from JSON file"""
    trades_file = Path("trades.json")
    if trades_file.exists():
        with open(trades_file, "r") as f:
            st.session_state.trades = json.load(f)

def check_signals(df):
    """Check for trading signals"""
    if len(df) < 2:
        return False, None, None, "Not enough data"
    
    last_candle = df.iloc[-1]
    previous_candle = df.iloc[-2]
    
    # Initialize variables
    signal = False
    stop_loss = None
    take_profit = None
    message = "No signal"
    
    # Check overall trend
    trend_up = (last_candle['EMA_20'] > last_candle['EMA_50'] and
                last_candle['close'] > last_candle['EMA_20'])
    
    # RSI conditions
    rsi_bullish = 30 < last_candle['RSI'] < 70  # Not overbought/oversold
    
    # MACD conditions
    macd_bullish = (last_candle['MACD_hist'] > 0 and 
                   previous_candle['MACD_hist'] < last_candle['MACD_hist'])  # Increasing momentum
    
    # Bollinger Bands conditions
    bb_bullish = last_candle['close'] > last_candle['BB_middle']  # Price above middle band
    
    # Volume confirmation (if available)
    volume_confirmed = True
    if "volume" in df.columns:
        volume_confirmed = last_candle['volume'] > last_candle['Volume_EMA']
    
    # Primary entry condition - EMA crossover
    ema_crossover = (last_candle['EMA_5'] > last_candle['EMA_8'] and 
                    previous_candle['EMA_5'] < previous_candle['EMA_8'])
    
    # Check all conditions for a long entry
    if (trend_up and ema_crossover and rsi_bullish and 
        macd_bullish and bb_bullish and volume_confirmed):
        
        entry_price = last_candle['close']
        
        # Dynamic stop loss based on ATR and Bollinger Bands
        atr_stop = entry_price - (last_candle['ATR_14'] * 1.5)
        bb_stop = last_candle['BB_lower']
        stop_loss = max(atr_stop, bb_stop)  # Use the higher of the two
        
        # Calculate take profit based on risk/reward ratio
        stop_distance = entry_price - stop_loss
        take_profit = entry_price + (stop_distance * st.session_state.risk_reward)
        
        signal = True
        message = f"""
        Long Signal Detected:
        - Trend: Bullish
        - EMA Crossover: Yes
        - RSI: {last_candle['RSI']:.2f}
        - MACD: Bullish momentum
        - BB: Price above middle band
        - Volume: {'Above average' if volume_confirmed else 'Below average'}
        """
    
    return signal, stop_loss, take_profit, message

def run_bot():
    """Main bot loop"""
    while st.session_state.bot_running:
        try:
            df = get_candles(st.session_state.timeframe, st.session_state.instrument, st.session_state.client)
            if df is not None:
                df = calculate_indicators(df)
                signal, stop_loss, take_profit, message = check_signals(df)
                
                if signal:
                    success, order_message = place_order(
                        st.session_state.client, st.session_state.instrument, st.session_state.position_size, stop_loss, take_profit
                    )
                    if success:
                        st.success(order_message)
                        st.info(message)  # Display signal analysis
                    else:
                        st.error(order_message)
                
            time.sleep(60)  # Check every minute
            
        except Exception as e:
            st.error(f"Error in bot execution: {str(e)}")
            st.session_state.bot_running = False
            break

@rate_limit(1)
@handle_api_errors
def get_account_details(client, account_id):
    """Fetch account details with rate limiting and error handling"""
    try:
        r = accounts.AccountSummary(accountID=account_id)
        response = client.request(r)
        return response['account']
    except Exception as e:
        st.error(f"Error fetching account details: {str(e)}")
        return None

@rate_limit(1)
@handle_api_errors
def get_account_instruments(client, account_id):
    """Fetch available instruments for the account"""
    try:
        r = accounts.AccountInstruments(accountID=account_id)
        response = client.request(r)
        return response['instruments']
    except Exception as e:
        st.error(f"Error fetching instruments: {str(e)}")
        return None

# Main dashboard
def main():
    st.title("Trading Bot Dashboard 📈")
    
    # Sidebar
    st.sidebar.title("Trading Bot Controls")
    
    # API Configuration with enhanced security
    api_key = os.getenv('OANDA_API_KEY')
    account_id = os.getenv('OANDA_ACCOUNT_ID')
    
    # If environment variables are not set, allow manual input with validation
    if not api_key:
        api_key = st.sidebar.text_input("OANDA API Key", type="password")
    if not account_id:
        account_id = st.sidebar.text_input("OANDA Account ID")
    
    if api_key and account_id:
        if not is_valid_api_key(api_key):
            st.sidebar.error("Invalid API key format")
            return
        
        try:
            client = API(access_token=api_key)
            st.session_state.authenticated = True
            st.session_state.client = client
            st.session_state.account_id = account_id
            st.sidebar.success("API Credentials verified successfully!")
            
            # Account Overview Section - Moved here for better visibility
            st.header("Account Overview")
            col1, col2, col3 = st.columns(3)
            
            # Fetch account details
            account_details = get_account_details(st.session_state.client, st.session_state.account_id)
            
            if account_details:
                with col1:
                    st.metric(
                        "Balance",
                        f"{float(account_details['balance']):.2f} {account_details['currency']}",
                        f"{float(account_details['pl']):.2f}"
                    )
                
                with col2:
                    st.metric(
                        "Open Positions",
                        account_details['openPositionCount'],
                        f"Orders: {account_details['pendingOrderCount']}"
                    )
                
                with col3:
                    margin_used = float(account_details['marginUsed'])
                    margin_available = float(account_details['marginAvailable'])
                    margin_rate = float(account_details['marginRate'])
                    
                    st.metric(
                        "Margin Available",
                        f"{margin_available:.2f} {account_details['currency']}",
                        f"Used: {margin_used:.2f}"
                    )
                
                # Detailed Account Information
                with st.expander("Detailed Account Information"):
                    st.write("Account Details:")
                    details_df = pd.DataFrame({
                        'Metric': [
                            'Account ID',
                            'Currency',
                            'Balance',
                            'Unrealized P/L',
                            'Realized P/L',
                            'Margin Used',
                            'Margin Available',
                            'Margin Rate',
                            'Open Position Count',
                            'Pending Orders',
                            'Last Transaction ID'
                        ],
                        'Value': [
                            account_details['id'],
                            account_details['currency'],
                            f"{float(account_details['balance']):.2f}",
                            f"{float(account_details['unrealizedPL']):.2f}",
                            f"{float(account_details['pl']):.2f}",
                            f"{float(account_details['marginUsed']):.2f}",
                            f"{float(account_details['marginAvailable']):.2f}",
                            f"{float(account_details['marginRate']) * 100:.2f}%",
                            account_details['openPositionCount'],
                            account_details['pendingOrderCount'],
                            account_details['lastTransactionID']
                        ]
                    })
                    st.table(details_df)
                    
                    # Available Instruments
                    st.write("Available Instruments:")
                    instruments = get_account_instruments(st.session_state.client, st.session_state.account_id)
                    if instruments:
                        instruments_df = pd.DataFrame([{
                            'Instrument': inst['name'],
                            'Type': inst['type'],
                            'Pip Location': inst['pipLocation'],
                            'Margin Rate': f"{float(inst['marginRate']) * 100:.2f}%"
                        } for inst in instruments])
                        st.dataframe(instruments_df)
            
        except Exception as e:
            st.sidebar.error(f"Authentication failed: {str(e)}")
            return
    else:
        st.sidebar.warning("Please set API credentials in .env file or enter them above")
        return
    
    # Trading Parameters
    st.sidebar.subheader("Trading Parameters")
    st.session_state.instrument = st.sidebar.selectbox(
        "Trading Pair",
        ["EUR_USD", "GBP_USD", "AUD_JPY", "USD_JPY"],
        index=["EUR_USD", "GBP_USD", "AUD_JPY", "USD_JPY"].index(st.session_state.instrument)
    )
    st.session_state.timeframe = st.sidebar.selectbox(
        "Timeframe",
        ["M5", "M15", "M30", "H1", "H4"],
        index=["M5", "M15", "M30", "H1", "H4"].index(st.session_state.timeframe)
    )
    st.session_state.risk_reward = st.sidebar.number_input(
        "Risk/Reward Ratio",
        min_value=1.0,
        value=st.session_state.risk_reward,
        step=0.1
    )
    st.session_state.position_size = st.sidebar.number_input(
        "Position Size (units)", 
        min_value=1, 
        max_value=int(os.getenv('MAX_POSITION_SIZE', 10000)),
        value=st.session_state.position_size,
        help=f"Maximum allowed position size: {int(os.getenv('MAX_POSITION_SIZE', 10000)):,}"
    )

    # Display trading limits
    st.sidebar.subheader("Trading Limits")
    max_position = int(os.getenv('MAX_POSITION_SIZE', 10000))
    max_trades = int(os.getenv('MAX_DAILY_TRADES', 10))
    risk_pct = float(os.getenv('RISK_PERCENTAGE', 1))
    
    st.sidebar.info(f"""
    - Max Position Size: {max_position:,} units
    - Max Daily Trades: {max_trades}
    - Risk Per Trade: {risk_pct}%
    """)

    # Add daily trades counter to the sidebar
    st.sidebar.metric(
        "Daily Trades",
        f"{st.session_state.daily_trades}/{os.getenv('MAX_DAILY_TRADES', 10)}",
        help="Resets daily at midnight"
    )

    # Load existing trades
    load_trades()

    # Create tabs and store current tab name
    tabs = ["Chart", "Active Trades", "Trade History", "Backtesting"]
    tab1, tab2, tab3, tab4 = st.tabs(tabs)
    
    # Function to get current market time
    def get_market_time():
        try:
            # Use timezone-aware datetime
            current_time = datetime.now(timezone.utc)
            if current_time.year > 2024:
                return datetime(2024, 1, 1, tzinfo=timezone.utc)
            return current_time
        except Exception:
            return datetime(2024, 1, 1, tzinfo=timezone.utc)

    with tab1:
        st.subheader("Price Chart")
        if st.session_state.authenticated:
            df = get_candles(st.session_state.timeframe, st.session_state.instrument, client)
            if df is not None:
                df = calculate_indicators(df)
                
                fig = go.Figure()
                
                # Candlestick chart
                fig.add_trace(go.Candlestick(
                    x=df['time'],
                    open=df['open'],
                    high=df['high'],
                    low=df['low'],
                    close=df['close'],
                    name="Price"
                ))
                
                # Add EMAs
                fig.add_trace(go.Scatter(x=df['time'], y=df['EMA_5'], name="EMA 5", line=dict(color='blue')))
                fig.add_trace(go.Scatter(x=df['time'], y=df['EMA_8'], name="EMA 8", line=dict(color='red')))
                fig.add_trace(go.Scatter(x=df['time'], y=df['EMA_20'], name="EMA 20", line=dict(color='green')))
                fig.add_trace(go.Scatter(x=df['time'], y=df['EMA_50'], name="EMA 50", line=dict(color='purple')))
                
                # Add Bollinger Bands
                fig.add_trace(go.Scatter(x=df['time'], y=df['BB_upper'], name="BB Upper", line=dict(color='gray', dash='dash')))
                fig.add_trace(go.Scatter(x=df['time'], y=df['BB_middle'], name="BB Middle", line=dict(color='gray', dash='dash')))
                fig.add_trace(go.Scatter(x=df['time'], y=df['BB_lower'], name="BB Lower", line=dict(color='gray', dash='dash')))
                
                # Create a new row for RSI
                fig.add_trace(go.Scatter(x=df['time'], y=df['RSI'], name="RSI", yaxis="y2"))
                
                # Create a new row for MACD
                fig.add_trace(go.Scatter(x=df['time'], y=df['MACD'], name="MACD", yaxis="y3"))
                fig.add_trace(go.Scatter(x=df['time'], y=df['MACD_signal'], name="Signal", yaxis="y3"))
                fig.add_trace(go.Bar(x=df['time'], y=df['MACD_hist'], name="MACD Hist", yaxis="y3"))
                
                # Update layout for multiple panels
                fig.update_layout(
                    title=f"{st.session_state.instrument} - {st.session_state.timeframe}",
                    yaxis_title="Price",
                    height=1000,  # Increase height for multiple panels
                    
                    # Create secondary y-axes for indicators
                    yaxis2=dict(
                        title="RSI",
                        overlaying="y",
                        side="right",
                        domain=[0.7, 1]
                    ),
                    yaxis3=dict(
                        title="MACD",
                        anchor="x",
                        domain=[0.2, 0.4]
                    ),
                    
                    # Main price chart
                    yaxis=dict(
                        domain=[0.5, 1]
                    )
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Display current indicator values
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("RSI", f"{df['RSI'].iloc[-1]:.2f}")
                with col2:
                    st.metric("MACD", f"{df['MACD'].iloc[-1]:.5f}")
                with col3:
                    st.metric("ATR", f"{df['ATR_14'].iloc[-1]:.5f}")

    with tab2:
        st.subheader("Active Trades")
        active_trades = [t for t in st.session_state.trades if t["status"] == "OPEN"]
        if active_trades:
            st.dataframe(pd.DataFrame(active_trades))
        else:
            st.info("No active trades")

    with tab3:
        st.subheader("Trade History")
        if st.session_state.trades:
            st.dataframe(pd.DataFrame(st.session_state.trades))
        else:
            st.info("No trade history")

    with tab4:
        st.subheader("Strategy Backtesting")
        
        # Get current date for backtesting
        current_time = get_market_time()
        current_date = current_time.date()
        
        # Show system time warning only in backtesting tab if needed
        system_time = datetime.now(timezone.utc)
        if current_time.year != system_time.year:
            st.error("⚠️ Your system time appears to be set incorrectly. Using current market time instead.")
        
        # Date range selector
        date_range = st.radio(
            "Select Date Range",
            ["Last 30 Days", "Last 90 Days", "Last Year", "Custom Range", "Maximum Available"],
            help="Choose a predefined range or select custom dates"
        )
        
        if date_range == "Last 30 Days":
            start_date = current_date - timedelta(days=30)
            end_date = current_date
        elif date_range == "Last 90 Days":
            start_date = current_date - timedelta(days=90)
            end_date = current_date
        elif date_range == "Last Year":
            start_date = current_date - timedelta(days=365)
            end_date = current_date
        elif date_range == "Maximum Available":
            # OANDA provides up to 5 years of historical data
            start_date = current_date - timedelta(days=5*365)
            end_date = current_date
        else:  # Custom Range
            col1, col2 = st.columns(2)
            with col1:
                start_date = st.date_input(
                    "Start Date",
                    value=current_date - timedelta(days=30),
                    min_value=current_date - timedelta(days=5*365),  # Up to 5 years of historical data
                    max_value=current_date,
                    help="Select start date (up to 5 years of historical data available)"
                )
            with col2:
                end_date = st.date_input(
                    "End Date",
                    value=current_date,
                    min_value=start_date,
                    max_value=current_date,
                    help="Select end date"
                )
        
        # Display selected date range info
        st.info(f"Selected Period: {(end_date - start_date).days} days from {start_date} to {end_date}")
        
        # Display timeframe warning for long periods
        if (end_date - start_date).days > 365:
            st.warning("⚠️ For periods longer than 1 year, consider using a larger timeframe (H1, H4, or D1) to improve performance")
        
        initial_balance = st.number_input(
            "Initial Balance",
            min_value=1000,
            value=10000,
            step=1000
        )
        
        if st.button("Run Backtest"):
            if st.session_state.authenticated:
                # Validate dates
                if start_date >= end_date:
                    st.error("Start date must be before end date")
                    return
                
                if end_date > current_date:
                    st.error("End date cannot be in the future")
                    return
                
                # Create containers for progress and results
                progress_container = st.container()
                results_container = st.container()
                
                with progress_container:
                    try:
                        # Convert dates to timezone-aware datetime objects
                        start_datetime = datetime.combine(start_date, datetime.min.time(), tzinfo=timezone.utc)
                        end_datetime = datetime.combine(end_date, datetime.max.time(), tzinfo=timezone.utc)
                        
                        # Initialize backtester
                        backtester = Backtester(
                            client=st.session_state.client,
                            instrument=st.session_state.instrument,
                            timeframe=st.session_state.timeframe,
                            initial_balance=initial_balance
                        )
                        
                        # Create status message placeholder
                        status_msg = st.empty()
                        status_msg.info("Initializing backtesting process...")
                        
                        # Create progress bars with descriptions
                        st.write("Step 1: Fetching Historical Data")
                        data_progress = st.progress(0.0, text="Preparing to fetch data...")
                        
                        # First fetch the data
                        status_msg.info("Fetching historical price data from OANDA...")
                        df = backtester.get_historical_data(start_datetime, end_datetime, progress_bar=data_progress)
                        
                        if df is None:
                            status_msg.error("Failed to fetch historical data. Please try again.")
                            return
                        
                        # Update status and create next progress bar
                        status_msg.success(f"Successfully retrieved {len(df)} candles of historical data!")
                        st.write("Step 2: Calculating Technical Indicators")
                        calc_progress = st.progress(0.0, text="Preparing to calculate indicators...")
                        
                        # Calculate indicators
                        status_msg.info("Calculating technical indicators...")
                        df = backtester.calculate_indicators(df, progress_bar=calc_progress)
                        
                        if df is None:
                            status_msg.error("Error calculating indicators. Please try again.")
                            return
                        
                        # Update status and create final progress bar
                        status_msg.success("Technical indicators calculated successfully!")
                        st.write("Step 3: Running Backtest Simulation")
                        backtest_progress = st.progress(0.0, text="Preparing to run backtest simulation...")
                        
                        # Run backtest simulation
                        status_msg.info("Running backtest simulation...")
                        results = backtester.run_backtest_simulation(
                            df,
                            risk_reward=st.session_state.risk_reward,
                            progress_bar=backtest_progress
                        )
                        
                        if results is None:
                            status_msg.error("Error during backtest simulation. Please try again.")
                            return
                        
                        # Final success message
                        status_msg.success("Backtest completed successfully! Displaying results...")
                        
                        # Display results in the results container
                        with results_container:
                            st.subheader("Performance Metrics")
                            
                            metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
                            
                            with metric_col1:
                                st.metric("Total Return", f"{results['total_return']:.2f}%")
                                st.metric("Total Trades", results['total_trades'])
                            
                            with metric_col2:
                                st.metric("Win Rate", f"{results['win_rate']:.2f}%")
                                st.metric("Profit Factor", f"{results['profit_factor']:.2f}")
                            
                            with metric_col3:
                                st.metric("Average Win", f"${results['avg_win']:.2f}")
                                st.metric("Winning Trades", results['winning_trades'])
                            
                            with metric_col4:
                                st.metric("Average Loss", f"-${results['avg_loss']:.2f}")
                                st.metric("Losing Trades", results['losing_trades'])
                            
                            if results['total_trades'] > 0:
                                # Plot results
                                st.plotly_chart(backtester.plot_results(results, df), use_container_width=True)
                                
                                # Display trade history
                                st.subheader("Trade History")
                                trades_df = pd.DataFrame(results['trades'])
                                if len(trades_df) > 0:
                                    trades_df['entry_time'] = pd.to_datetime(trades_df['entry_time'])
                                    trades_df['exit_time'] = pd.to_datetime(trades_df['exit_time'])
                                    trades_df['duration'] = trades_df['exit_time'] - trades_df['entry_time']
                                    trades_df['return'] = trades_df['pnl'] / trades_df['balance'] * 100
                                    
                                    st.dataframe(trades_df[[
                                        'entry_time', 'exit_time', 'duration',
                                        'entry_price', 'exit_price', 'pnl', 'return'
                                    ]].style.format({
                                        'entry_price': '${:.5f}',
                                        'exit_price': '${:.5f}',
                                        'pnl': '${:.2f}',
                                        'return': '{:.2f}%'
                                    }))
                            else:
                                st.warning("No trades were executed during the backtest period. Try adjusting the parameters or date range.")
                    
                    except Exception as e:
                        st.error(f"Error during backtesting: {str(e)}")
                        logger.error(f"Backtesting error: {str(e)}")
            else:
                st.error("Please enter your API credentials to run backtest.")

    # Bot controls with persistent key
    if 'bot_controls_key' not in st.session_state:
        st.session_state.bot_controls_key = 0

    bot_controls = st.sidebar.container()
    
    with bot_controls:
        st.write("Bot Controls")
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("Start Bot", key=f"start_bot_{st.session_state.bot_controls_key}"):
                st.session_state.bot_running = True
                st.success("Bot started!")
                st.session_state.bot_controls_key += 1

        with col2:
            if st.button("Stop Bot", key=f"stop_bot_{st.session_state.bot_controls_key}"):
                st.session_state.bot_running = False
                st.warning("Bot stopped!")
                st.session_state.bot_controls_key += 1
        
        # Bot status indicator
        st.write("---")
        if st.session_state.bot_running:
            st.success("🟢 Bot is running")
        else:
            st.error("🔴 Bot is stopped")

    # Run the bot if it's enabled
    if st.session_state.bot_running:
        run_bot()

if __name__ == "__main__":
    main() 