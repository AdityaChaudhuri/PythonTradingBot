# Python Forex Trading Bot

## Overview
An automated forex trading system built with Python that integrates real-time market analysis and automated trade execution through the OANDA API. The system features a Streamlit-based dashboard for monitoring trades and analyzing market data.

## Features
- Real-time forex data processing using OANDA API
- Technical analysis with multiple indicators (EMA, RSI, MACD, Bollinger Bands)
- Automated trade execution with risk management
- Interactive web dashboard built with Streamlit
- Dynamic position sizing and stop-loss mechanisms
- Backtesting capabilities for strategy optimization

## Tech Stack
- Python 3.x
- OANDA API for forex data and trading
- Streamlit for web interface
- Pandas & NumPy for data processing
- pandas-ta for technical indicators
- Plotly for interactive charts

## Project Structure
```
trading_bot/
├── src/
│   ├── api/
│   │   ├── __init__.py
│   │   └── oanda_api.py
│   ├── analysis/
│   │   ├── __init__.py
│   │   └── indicators.py
│   ├── trading/
│   │   ├── __init__.py
│   │   ├── strategy.py
│   │   └── risk_management.py
│   └── visualization/
│       ├── __init__.py
│       └── dashboard.py
├── config/
│   └── config.yaml
├── documentation/
│   ├── trading_guide.md
│   └── api_documentation.md
├── tests/
│   └── test_trading.py
├── requirements.txt
└── README.md
```

## Installation
1. Clone the repository:
```bash
git clone https://github.com/AdityaChaudhuri/PythonTradingBot.git
cd PythonTradingBot
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Configure OANDA API credentials:
- Create a config.yaml file in the config directory
- Add your OANDA API key and account ID

## Usage
1. Start the Streamlit dashboard:
```bash
streamlit run src/visualization/dashboard.py
```

2. Access the dashboard at http://localhost:8501

## Features in Detail

### Technical Analysis
- Multiple timeframe analysis
- Support for various technical indicators:
  - Exponential Moving Averages (EMA)
  - Relative Strength Index (RSI)
  - Moving Average Convergence Divergence (MACD)
  - Bollinger Bands

### Risk Management
- Dynamic position sizing based on account balance
- Automated stop-loss calculation
- Maximum position size limits
- Risk percentage per trade configuration

### Dashboard Features
- Real-time price charts
- Technical indicator visualization
- Trade execution interface
- Performance metrics
- Portfolio monitoring

## Contributing
Contributions are welcome! Please feel free to submit a Pull Request.

## License
This project is licensed under the MIT License - see the LICENSE file for details.

## Disclaimer
This trading bot is for educational purposes only. Trading forex carries significant risks, and you should carefully consider whether trading is appropriate for you based on your financial condition.

## Contact
Aditya Narayan Chaudhuri - aditya.nc.552@email.com 