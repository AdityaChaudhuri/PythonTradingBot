import pandas as pd
import pandas_ta as ta
import numpy as np
from datetime import datetime, timedelta, timezone
from oandapyV20 import API
import oandapyV20.endpoints.instruments as instruments
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
import streamlit as st

class Backtester:
    def __init__(self, client, instrument, timeframe, initial_balance=10000):
        self.client = client
        self.instrument = instrument
        self.timeframe = timeframe
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.trades = []
        self.current_position = None
    
    def get_historical_data(self, start_date, end_date=None, progress_bar=None):
        """Fetch historical data from OANDA using pagination"""
        try:
            # Validate and adjust dates
            current_time = datetime.now(timezone.utc)
            
            # Ensure start_date is timezone-aware
            if start_date.tzinfo is None:
                start_date = start_date.replace(tzinfo=timezone.utc)
            
            # If end_date is not provided or is in the future, use current time
            if end_date is None:
                end_date = current_time
            elif end_date.tzinfo is None:
                end_date = end_date.replace(tzinfo=timezone.utc)
            
            if end_date > current_time:
                print("End date cannot be in the future, adjusting to current time")
                end_date = current_time
            
            # Ensure start_date is not in the future
            if start_date > current_time:
                print("Start date cannot be in the future, adjusting to 30 days before current time")
                start_date = current_time - timedelta(days=30)
            
            # Ensure start_date is before end_date
            if start_date >= end_date:
                print("Start date must be before end date, adjusting dates")
                start_date = end_date - timedelta(days=30)
            
            print(f"\nFetching data from {start_date} to {end_date}")
            if progress_bar:
                progress_bar.progress(0.0, text=f"Starting data fetch from {start_date.date()} to {end_date.date()}")
            
            # Initialize empty list for all candles
            all_candles = []
            current_start = start_date
            
            # For M15, we'll use 3-day chunks to ensure we get enough candles per request
            chunk_size = timedelta(days=3)
            
            # Calculate total time span
            total_days = (end_date - start_date).days
            days_processed = 0
            
            print(f"Using {chunk_size.days} day chunks for {self.timeframe} timeframe")
            
            while current_start < end_date:
                # Calculate the end time for this chunk
                chunk_end = min(current_start + chunk_size, end_date)
                
                # Update progress
                if progress_bar is not None:
                    progress = min(days_processed / total_days, 0.99) if total_days > 0 else 0.99
                    progress_bar.progress(progress, text=f"Fetching data: {days_processed}/{total_days} days processed")
                
                # Format dates in OANDA's required format
                params = {
                    "granularity": self.timeframe,
                    "from": current_start.strftime("%Y-%m-%dT%H:%M:%SZ"),
                    "to": chunk_end.strftime("%Y-%m-%dT%H:%M:%SZ"),
                    "price": "M"
                }
                
                try:
                    print(f"Requesting {self.timeframe} candles from {current_start.date()} to {chunk_end.date()}")
                    r = instruments.InstrumentsCandles(instrument=self.instrument, params=params)
                    response = self.client.request(r)
                    chunk_candles = response.get('candles', [])
                    
                    if not chunk_candles:
                        print(f"No candles received for period {current_start.date()} to {chunk_end.date()}")
                        # Move to next chunk if no data
                        current_start = chunk_end
                        days_processed += chunk_size.days
                        continue
                    
                    # Process candles
                    new_candles = []
                    for candle in chunk_candles:
                        if candle["complete"]:
                            timestamp = candle["time"].split('.')[0]
                            candle_time = datetime.strptime(timestamp, "%Y-%m-%dT%H:%M:%S").replace(tzinfo=timezone.utc)
                            
                            new_candles.append({
                                "time": candle_time,
                                "open": float(candle["mid"]["o"]),
                                "high": float(candle["mid"]["h"]),
                                "low": float(candle["mid"]["l"]),
                                "close": float(candle["mid"]["c"]),
                                "volume": int(candle["volume"])
                            })
                    
                    all_candles.extend(new_candles)
                    print(f"Received {len(new_candles)} candles for {current_start.date()} to {chunk_end.date()}")
                    
                    # Always move forward by chunk_size to avoid getting stuck
                    current_start = chunk_end
                    days_processed += chunk_size.days
                    
                    time.sleep(0.5)  # Rate limiting
                    
                except Exception as chunk_error:
                    print(f"Error fetching chunk: {str(chunk_error)}")
                    # Move to next chunk on error
                    current_start = chunk_end
                    days_processed += chunk_size.days
            
            # Set progress to 100% when done
            if progress_bar is not None:
                progress_bar.progress(1.0, text=f"Data fetching complete! Retrieved {len(all_candles)} candles")
            
            if not all_candles:
                print("No data available for the specified period")
                return None
            
            print(f"Total candles retrieved: {len(all_candles)}")
            
            # Create DataFrame and sort
            df = pd.DataFrame(all_candles)
            df = df.sort_values('time')
            df = df.drop_duplicates(subset=['time'])
            
            # Calculate basic volume MA for signal checking
            df['volume_ma'] = df['volume'].rolling(window=20).mean()
            
            print(f"Data processing complete. Final dataset has {len(df)} candles")
            return df
        
        except Exception as e:
            print(f"Critical error in get_historical_data: {str(e)}")
            if progress_bar is not None:
                progress_bar.error(f"Error fetching data: {str(e)}")
            return None
    
    def calculate_indicators(self, df, progress_bar=None):
        """Calculate technical indicators with progress tracking"""
        try:
            print("\nStarting indicator calculations...")
            st.write("Starting indicator calculations...")  # Add visual feedback
            
            if df is None or len(df) == 0:
                print("Error: Empty DataFrame provided to calculate_indicators")
                return None

            # Make a copy and verify data
            df = df.copy()
            print(f"DataFrame shape: {df.shape}")
            st.write(f"Processing {df.shape[0]} candles...")  # Add visual feedback
            
            total_steps = 8
            current_step = 0
            
            # Trend Indicators - EMAs
            print("\nCalculating EMAs...")
            st.write("Calculating EMAs...")  # Add visual feedback
            current_step += 1
            if progress_bar is not None:
                progress_bar.progress(current_step/total_steps, text="Calculating EMAs...")
            
            try:
                for period in [5, 8, 20, 50]:
                    df[f"EMA_{period}"] = df["close"].ewm(span=period, adjust=False).mean()
                    st.write(f"EMA {period} calculated")  # Add visual feedback
                print("EMAs calculated successfully")
            except Exception as e:
                print(f"Error calculating EMAs: {str(e)}")
                return None
            
            # ATR
            print("\nCalculating ATR...")
            st.write("Calculating ATR...")  # Add visual feedback
            current_step += 1
            if progress_bar is not None:
                progress_bar.progress(current_step/total_steps, text="Calculating ATR...")
            
            try:
                high_low = df['high'] - df['low']
                high_close = abs(df['high'] - df['close'].shift())
                low_close = abs(df['low'] - df['close'].shift())
                ranges = pd.concat([high_low, high_close, low_close], axis=1)
                true_range = ranges.max(axis=1)
                df["ATR_14"] = true_range.rolling(window=14).mean()
                st.write("ATR calculated successfully")  # Add visual feedback
            except Exception as e:
                print(f"Error calculating ATR: {str(e)}")
                return None
            
            # Bollinger Bands
            print("\nCalculating Bollinger Bands...")
            current_step += 1
            if progress_bar is not None:
                try:
                    progress_bar.progress(current_step/total_steps, text="Calculating Bollinger Bands...")
                except Exception as e:
                    print(f"Error updating BB progress: {str(e)}")
            
            try:
                df["BB_middle"] = df["close"].rolling(window=20).mean()
                rolling_std = df["close"].rolling(window=20).std()
                df["BB_upper"] = df["BB_middle"] + (rolling_std * 2)
                df["BB_lower"] = df["BB_middle"] - (rolling_std * 2)
                print("Bollinger Bands calculated successfully")
            except Exception as e:
                print(f"Error calculating Bollinger Bands: {str(e)}")
                return None
            
            # RSI
            print("\nCalculating RSI...")
            current_step += 1
            if progress_bar is not None:
                try:
                    progress_bar.progress(current_step/total_steps, text="Calculating RSI...")
                except Exception as e:
                    print(f"Error updating RSI progress: {str(e)}")
            
            try:
                delta = df["close"].diff()
                gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
                rs = gain / loss
                df["RSI"] = 100 - (100 / (1 + rs))
                print("RSI calculated successfully")
            except Exception as e:
                print(f"Error calculating RSI: {str(e)}")
                return None
            
            # MACD
            print("\nCalculating MACD...")
            current_step += 1
            if progress_bar is not None:
                try:
                    progress_bar.progress(current_step/total_steps, text="Calculating MACD...")
                except Exception as e:
                    print(f"Error updating MACD progress: {str(e)}")
            
            try:
                exp1 = df["close"].ewm(span=12, adjust=False).mean()
                exp2 = df["close"].ewm(span=26, adjust=False).mean()
                df["MACD"] = exp1 - exp2
                df["MACD_signal"] = df["MACD"].ewm(span=9, adjust=False).mean()
                df["MACD_hist"] = df["MACD"] - df["MACD_signal"]
                print("MACD calculated successfully")
            except Exception as e:
                print(f"Error calculating MACD: {str(e)}")
                return None
            
            # Volume
            print("\nCalculating Volume indicators...")
            current_step += 1
            if progress_bar is not None:
                try:
                    progress_bar.progress(current_step/total_steps, text="Calculating Volume indicators...")
                except Exception as e:
                    print(f"Error updating Volume progress: {str(e)}")
            
            try:
                if "volume" in df.columns:
                    df["Volume_EMA"] = df["volume"].ewm(span=20, adjust=False).mean()
                else:
                    print("No volume data available, using zeros")
                    df["volume"] = 0
                    df["Volume_EMA"] = 0
                print("Volume indicators calculated successfully")
            except Exception as e:
                print(f"Error calculating Volume indicators: {str(e)}")
                return None
            
            # Data Cleaning
            print("\nCleaning data...")
            current_step += 1
            if progress_bar is not None:
                try:
                    progress_bar.progress(current_step/total_steps, text="Cleaning data...")
                except Exception as e:
                    print(f"Error updating cleaning progress: {str(e)}")
            
            try:
                df = df.fillna(method='ffill')
                df = df.fillna(method='bfill')
                df = df.fillna(0)  # Fill any remaining NaNs with 0
                print("Data cleaning completed successfully")
            except Exception as e:
                print(f"Error during data cleaning: {str(e)}")
                return None
            
            # Final verification
            print("\nVerifying calculations...")
            current_step += 1
            if progress_bar is not None:
                try:
                    progress_bar.progress(current_step/total_steps, text="Verifying calculations...")
                except Exception as e:
                    print(f"Error updating verification progress: {str(e)}")
            
            required_columns = [
                "EMA_5", "EMA_8", "EMA_20", "EMA_50",
                "ATR_14",
                "BB_upper", "BB_middle", "BB_lower",
                "RSI",
                "MACD", "MACD_signal", "MACD_hist",
                "volume", "Volume_EMA"
            ]
            
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                print(f"Missing columns: {missing_columns}")
                return None
            
            if progress_bar is not None:
                try:
                    progress_bar.progress(1.0, text="Indicator calculations complete!")
                except Exception as e:
                    print(f"Error updating final progress: {str(e)}")
            
            print("\nAll calculations completed successfully!")
            st.write("All indicators calculated successfully!")  # Add visual feedback
            return df
            
        except Exception as e:
            print(f"Critical error in calculate_indicators: {str(e)}")
            st.error(f"Error calculating indicators: {str(e)}")  # Add visual feedback
            return None
    
    def check_signals(self, row, prev_row):
        """Check for trading signals"""
        try:
            # Check overall trend
            trend_up = (row['EMA_20'] > row['EMA_50'] and
                       row['close'] > row['EMA_20'])
            
            # RSI conditions
            rsi_bullish = 30 < row['RSI'] < 70
            
            # MACD conditions
            macd_bullish = (row['MACD_hist'] > 0 and 
                           prev_row['MACD_hist'] < row['MACD_hist'])
            
            # Bollinger Bands conditions
            bb_bullish = row['close'] > row['BB_middle']
            
            # Volume confirmation
            volume_confirmed = row['volume'] > row['volume_ma']
            
            # EMA crossover
            ema_crossover = (row['EMA_5'] > row['EMA_8'] and 
                            prev_row['EMA_5'] < prev_row['EMA_8'])
            
            return (trend_up and ema_crossover and rsi_bullish and 
                    macd_bullish and bb_bullish and volume_confirmed)
        except Exception as e:
            print(f"Error checking signals: {str(e)}")
            return False
    
    def run_backtest(self, start_date, end_date=None, risk_reward=1.5, progress_container=None):
        """Run backtest over specified period"""
        try:
            # Create progress bars if container provided
            if progress_container is not None:
                data_progress = progress_container.progress(0.0, text="Fetching historical data...")
                calc_progress = progress_container.progress(0.0, text="Calculating indicators...")
                backtest_progress = progress_container.progress(0.0, text="Waiting to start backtest...")
            else:
                data_progress = None
                calc_progress = None
                backtest_progress = None
            
            # Get historical data with progress tracking
            df = self.get_historical_data(start_date, end_date, progress_bar=data_progress)
            
            if df is None or len(df) < 50:
                if progress_container is not None:
                    progress_container.error("Insufficient data for backtesting")
                return self._empty_results(), None
            
            # Calculate indicators with progress tracking
            df = self.calculate_indicators(df, progress_bar=calc_progress)
            
            if df is None:
                if progress_container is not None:
                    progress_container.error("Error calculating indicators")
                return self._empty_results(), None
            
            # Initialize results
            positions = []
            equity_curve = [self.initial_balance]
            self.trades = []
            self.balance = self.initial_balance
            self.current_position = None
            
            # Run backtest
            total_candles = len(df)
            for i in range(1, total_candles):
                if backtest_progress is not None and i % 100 == 0:  # Update progress every 100 candles
                    progress = i / total_candles
                    backtest_progress.progress(progress, text=f"Processing candle {i}/{total_candles}")
                
                current_price = df.iloc[i]['close']
                
                if self.current_position is None:
                    if self.check_signals(df.iloc[i], df.iloc[i-1]):
                        entry_price = current_price
                        stop_loss = entry_price - (df.iloc[i]['ATR_14'] * 1.5)
                        take_profit = entry_price + ((entry_price - stop_loss) * risk_reward)
                        
                        risk_amount = self.balance * 0.01
                        position_size = risk_amount / (entry_price - stop_loss)
                        
                        self.current_position = {
                            'entry_price': entry_price,
                            'stop_loss': stop_loss,
                            'take_profit': take_profit,
                            'size': position_size,
                            'entry_time': df.iloc[i]['time']
                        }
                        positions.append(self.current_position)
                
                elif self.current_position is not None:
                    if (current_price <= self.current_position['stop_loss'] or 
                        current_price >= self.current_position['take_profit']):
                        
                        pnl = (current_price - self.current_position['entry_price']) * self.current_position['size']
                        self.balance += pnl
                        
                        self.trades.append({
                            'entry_time': self.current_position['entry_time'],
                            'exit_time': df.iloc[i]['time'],
                            'entry_price': self.current_position['entry_price'],
                            'exit_price': current_price,
                            'pnl': pnl,
                            'balance': self.balance
                        })
                        
                        self.current_position = None
                
                equity_curve.append(self.balance)
            
            if backtest_progress is not None:
                backtest_progress.progress(1.0, text="Backtest complete!")
            
            # Calculate performance metrics
            trades_df = pd.DataFrame(self.trades)
            if len(trades_df) > 0:
                results = self._calculate_metrics(trades_df, equity_curve)
            else:
                results = self._empty_results()
            
            return results, df
        
        except Exception as e:
            print(f"Error in backtesting: {str(e)}")
            if progress_container is not None:
                progress_container.error(f"Error in backtesting: {str(e)}")
            return self._empty_results(), None
    
    def _empty_results(self):
        """Return empty results structure"""
        return {
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'win_rate': 0,
            'avg_win': 0,
            'avg_loss': 0,
            'profit_factor': 0,
            'total_return': 0,
            'equity_curve': [self.initial_balance],
            'trades': []
        }
    
    def _calculate_metrics(self, trades_df, equity_curve):
        """Calculate performance metrics from trades DataFrame"""
        win_rate = len(trades_df[trades_df['pnl'] > 0]) / len(trades_df) * 100
        avg_win = trades_df[trades_df['pnl'] > 0]['pnl'].mean() if len(trades_df[trades_df['pnl'] > 0]) > 0 else 0
        avg_loss = abs(trades_df[trades_df['pnl'] < 0]['pnl'].mean()) if len(trades_df[trades_df['pnl'] < 0]) > 0 else 0
        profit_factor = abs(avg_win / avg_loss) if avg_loss != 0 else float('inf')
        
        return {
            'total_trades': len(trades_df),
            'winning_trades': len(trades_df[trades_df['pnl'] > 0]),
            'losing_trades': len(trades_df[trades_df['pnl'] < 0]),
            'win_rate': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'total_return': (self.balance - self.initial_balance) / self.initial_balance * 100,
            'equity_curve': equity_curve,
            'trades': self.trades
        }
    
    def plot_results(self, results, df):
        """Plot backtest results"""
        fig = make_subplots(rows=3, cols=1, 
                           shared_xaxes=True,
                           vertical_spacing=0.05,
                           row_heights=[0.6, 0.2, 0.2])
        
        # Price chart with indicators
        fig.add_trace(go.Candlestick(
            x=df['time'],
            open=df['open'],
            high=df['high'],
            low=df['low'],
            close=df['close'],
            name="Price"
        ), row=1, col=1)
        
        # Add EMAs
        fig.add_trace(go.Scatter(x=df['time'], y=df['EMA_5'], name="EMA 5", line=dict(color='blue')), row=1, col=1)
        fig.add_trace(go.Scatter(x=df['time'], y=df['EMA_8'], name="EMA 8", line=dict(color='red')), row=1, col=1)
        fig.add_trace(go.Scatter(x=df['time'], y=df['EMA_20'], name="EMA 20", line=dict(color='green')), row=1, col=1)
        
        # Add trade markers
        for trade in results['trades']:
            # Entry marker
            fig.add_trace(go.Scatter(
                x=[trade['entry_time']],
                y=[trade['entry_price']],
                mode='markers',
                marker=dict(symbol='triangle-up', size=15, color='green'),
                name='Entry'
            ), row=1, col=1)
            
            # Exit marker
            fig.add_trace(go.Scatter(
                x=[trade['exit_time']],
                y=[trade['exit_price']],
                mode='markers',
                marker=dict(symbol='triangle-down', size=15, 
                          color='red' if trade['pnl'] < 0 else 'green'),
                name='Exit'
            ), row=1, col=1)
        
        # Equity curve
        fig.add_trace(go.Scatter(
            x=df['time'],
            y=results['equity_curve'],
            name="Equity Curve",
            line=dict(color='blue')
        ), row=2, col=1)
        
        # MACD
        fig.add_trace(go.Scatter(x=df['time'], y=df['MACD'], name="MACD", line=dict(color='blue')), row=3, col=1)
        fig.add_trace(go.Scatter(x=df['time'], y=df['MACD_signal'], name="Signal", line=dict(color='red')), row=3, col=1)
        fig.add_trace(go.Bar(x=df['time'], y=df['MACD_hist'], name="Histogram"), row=3, col=1)
        
        fig.update_layout(
            title="Backtest Results",
            xaxis_title="Date",
            yaxis_title="Price",
            height=1200
        )
        
        return fig

    def run_backtest_simulation(self, df, risk_reward=1.5, progress_bar=None):
        """Run backtest simulation on pre-calculated data"""
        try:
            if df is None or len(df) < 50:
                st.error("Insufficient data for backtesting")
                return None
            
            # Initialize results
            positions = []
            equity_curve = [self.initial_balance]
            self.trades = []
            self.balance = self.initial_balance
            self.current_position = None
            
            # Run backtest
            total_candles = len(df)
            st.write(f"Starting backtest simulation on {total_candles} candles...")
            
            for i in range(1, total_candles):
                if progress_bar is not None and i % 100 == 0:  # Update progress every 100 candles
                    progress = i / total_candles
                    progress_bar.progress(progress, text=f"Processing candle {i}/{total_candles}")
                
                current_price = df.iloc[i]['close']
                
                if self.current_position is None:
                    if self.check_signals(df.iloc[i], df.iloc[i-1]):
                        entry_price = current_price
                        stop_loss = entry_price - (df.iloc[i]['ATR_14'] * 1.5)
                        take_profit = entry_price + ((entry_price - stop_loss) * risk_reward)
                        
                        risk_amount = self.balance * 0.01
                        position_size = risk_amount / (entry_price - stop_loss)
                        
                        self.current_position = {
                            'entry_price': entry_price,
                            'stop_loss': stop_loss,
                            'take_profit': take_profit,
                            'size': position_size,
                            'entry_time': df.iloc[i]['time']
                        }
                        positions.append(self.current_position)
                
                elif self.current_position is not None:
                    if (current_price <= self.current_position['stop_loss'] or 
                        current_price >= self.current_position['take_profit']):
                        
                        pnl = (current_price - self.current_position['entry_price']) * self.current_position['size']
                        self.balance += pnl
                        
                        self.trades.append({
                            'entry_time': self.current_position['entry_time'],
                            'exit_time': df.iloc[i]['time'],
                            'entry_price': self.current_position['entry_price'],
                            'exit_price': current_price,
                            'pnl': pnl,
                            'balance': self.balance
                        })
                        
                        self.current_position = None
                
                equity_curve.append(self.balance)
            
            if progress_bar is not None:
                progress_bar.progress(1.0, text="Backtest complete!")
            
            # Calculate performance metrics
            trades_df = pd.DataFrame(self.trades)
            if len(trades_df) > 0:
                results = self._calculate_metrics(trades_df, equity_curve)
                st.write(f"Backtest completed with {len(trades_df)} trades")
            else:
                results = self._empty_results()
                st.write("No trades were executed during the backtest period")
            
            return results
        
        except Exception as e:
            print(f"Error in backtesting: {str(e)}")
            st.error(f"Error in backtesting: {str(e)}")
            return None 