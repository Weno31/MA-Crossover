import matplotlib.pyplot as plt
import yfinance as yf
import numpy as np
import pandas as pd
import streamlit as st
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

#Page configuration using streamlit
st.set_page_config(page_title='Moving Average Crossover Strategy', layout='wide', page_icon='📈')

#Setting options for indexes
INDEX_OPTIONS = {"S&P 500": "^GSPC", 
                 "Dow Jones Industrial Average": "^DJI",
                 "NASDAQ Composite": "^IXIC", "Russell 2000": "^RUT",
                 "FTSE 100": "^FTSE",
                 "DAX (Germany)": "^GDAXI",
                 "CAC 40 (France)": "^FCHI",
                 "Nikkei 225 (Japan)": "^N225",
                 "Hang Seng Index (Hong Kong)":
                 "^HSI", "Nifty 50 (India)": "^NSEI"}

#sidebar configuration
def setup_sidebar():
    """Setup sidebar with LinkedIn profile and strategy parameters"""
    # LinkedIn Profile Link
    st.sidebar.markdown("""
        <div style="text-align: center; padding: 20px 0;">
            <a href="https://www.linkedin.com/in/golesedimonngakgotla/" target="_blank" 
               style="text-decoration: none; color: #0077B5;">
                <img src="https://cdn-icons-png.flaticon.com/512/174/174857.png" 
                     width="25" height="25" style="margin-right: 8px;">
                <span style="vertical-align: middle;">LinkedIn</span>
            </a>
        </div>
    """, unsafe_allow_html=True)
    
    st.sidebar.title("📊 Strategy Parameters")
    
    # User inputs with validation
    index = st.sidebar.selectbox("Select an index:", list(INDEX_OPTIONS.keys()))
    
    col1, col2 = st.sidebar.columns(2)
    with col1:
        fast_ma = st.sidebar.number_input(
            "Fast MA period:", 
            min_value=5, 
            max_value=100, 
            value=50, 
            step=5
        )
    with col2:
        slow_ma = st.sidebar.number_input(
            "Slow MA period:", 
            min_value=50, 
            max_value=300, 
            value=200, 
            step=10
        )
    
    lookback = st.sidebar.number_input(
        "Lookback period (days):", 
        min_value=50, 
        max_value=10000, 
        value=1000, 
        step=100
    )
    
    # Validation
    if fast_ma >= slow_ma:
        st.sidebar.error("⚠️ Fast MA must be less than Slow MA")
        return None, None, None, None
    
    return index, fast_ma, slow_ma, lookback

@st.cache_data(ttl=3600)  # Cache for 1 hour
def get_data(ticker, lookback_days):
    """Fetch and process market data with error handling"""
    try:
        # Calculate start date
        end_date = datetime.now()
        start_date = end_date - timedelta(days=lookback_days + 100)
        
        df = yf.download(ticker, start=start_date, end=end_date, progress=False)
        
        if df.empty:
            st.error(f"No data available for {ticker}")
            return None
        
        # Handle multi-level columns if present
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        
        # Return only the requested lookback period
        return df.iloc[-lookback_days:].copy()
        
    except Exception as e:
        st.error(f"Error fetching data: {str(e)}")
        return None

def calculate_moving_averages(df, fast_period, slow_period):
    """Calculate moving averages and volume indicators"""
    df = df.copy()
    
    # Calculate moving averages
    df[f'MA_{fast_period}'] = df['Close'].rolling(window=fast_period, min_periods=1).mean()
    df[f'MA_{slow_period}'] = df['Close'].rolling(window=slow_period, min_periods=1).mean()
    df['Volume_MA'] = df['Volume'].rolling(window=20, min_periods=1).mean()
    
    return df

def generate_signals(df, fast_period, slow_period):
    """Generate trading signals based on MA crossover and volume"""
    df = df.copy()
    
    # Basic trend signal
    trend_signal = np.where(df[f'MA_{fast_period}'] > df[f'MA_{slow_period}'], 1, -1)
    
    # Volume condition (only trade when volume is above average)
    volume_condition = df['Volume'] > df['Volume_MA']
    
    # Combined signal
    df['Signal'] = np.where(volume_condition, trend_signal, 0)
    
    # Avoid lookahead bias
    df['Signal'] = df['Signal'].shift(1)
    
    return df.dropna()

def calculate_returns(df):
    """Calculate asset and strategy returns"""
    df = df.copy()
    
    # Calculate returns
    df['Asset_Returns'] = df['Close'].pct_change()
    df['Strategy_Returns'] = df['Asset_Returns'] * df['Signal']
    
    # Calculate cumulative returns
    df['Cumulative_Asset'] = (1 + df['Asset_Returns']).cumprod() - 1
    df['Cumulative_Strategy'] = (1 + df['Strategy_Returns']).cumprod() - 1
    
    return df

def calculate_metrics(df):
    """Calculate performance metrics"""
    asset_returns = df['Asset_Returns'].dropna()
    strategy_returns = df['Strategy_Returns'].dropna()
    
    metrics = {
        'Total Asset Return': f"{df['Cumulative_Asset'].iloc[-1]:.2%}",
        'Total Strategy Return': f"{df['Cumulative_Strategy'].iloc[-1]:.2%}",
        'Asset Volatility': f"{asset_returns.std() * np.sqrt(252):.2%}",
        'Strategy Volatility': f"{strategy_returns.std() * np.sqrt(252):.2%}",
        'Asset Sharpe Ratio': f"{(asset_returns.mean() / asset_returns.std()) * np.sqrt(252):.2f}",
        'Strategy Sharpe Ratio': f"{(strategy_returns.mean() / strategy_returns.std()) * np.sqrt(252):.2f}" if strategy_returns.std() != 0 else "N/A",
        'Max Drawdown (Asset)': f"{(df['Cumulative_Asset'] / df['Cumulative_Asset'].cummax() - 1).min():.2%}",
        'Max Drawdown (Strategy)': f"{(df['Cumulative_Strategy'] / df['Cumulative_Strategy'].cummax() - 1).min():.2%}",
    }
    
    return metrics

def create_price_chart(df, index_name, fast_period, slow_period):
    """Create price and moving averages chart"""
    fig, ax = plt.subplots(figsize=(14, 8))
    
    ax.plot(df.index, df['Close'], label='Close Price', linewidth=1.5, color='black')
    ax.plot(df.index, df[f'MA_{fast_period}'], label=f'{fast_period}-day MA', linewidth=1, color='blue')
    ax.plot(df.index, df[f'MA_{slow_period}'], label=f'{slow_period}-day MA', linewidth=1, color='red')
    
    ax.set_title(f'{index_name} - Price and Moving Averages', fontsize=16, fontweight='bold')
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Price', fontsize=12)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

def create_performance_chart(df, index_name):
    """Create performance comparison chart"""
    fig, ax = plt.subplots(figsize=(14, 8))
    
    ax.plot(df.index, df['Cumulative_Asset'] * 100, 
            label='Buy & Hold', linewidth=2, color='blue')
    ax.plot(df.index, df['Cumulative_Strategy'] * 100, 
            label='MA Crossover Strategy', linewidth=2, color='green')
    
    ax.set_title(f'{index_name} - Strategy Performance Comparison', fontsize=16, fontweight='bold')
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Cumulative Returns (%)', fontsize=12)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

def main():
    """Main application function"""
    st.title("📈 Moving Average Crossover Strategy Backtester")
    st.subheader("By: Golesedi W. Monngakgotla")
    st.markdown("---")
    
    # Setup sidebar and get parameters
    params = setup_sidebar()
    if params[0] is None:  # Validation failed
        st.stop()
    
    index_name, fast_ma, slow_ma, lookback = params
    ticker = INDEX_OPTIONS[index_name]
    
    # Main content
    with st.spinner('Fetching market data...'):
        df = get_data(ticker, lookback)
    
    if df is None:
        st.stop()
    
    # Process data
    with st.spinner('Calculating strategy...'):
        df = calculate_moving_averages(df, fast_ma, slow_ma)
        df = generate_signals(df, fast_ma, slow_ma)
        df = calculate_returns(df)
    
    # Display results
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader(f"📊 {index_name} Analysis")
        
        # Price chart
        fig1 = create_price_chart(df, index_name, fast_ma, slow_ma)
        st.pyplot(fig1)
        
        # Performance chart
        fig2 = create_performance_chart(df, index_name)
        st.pyplot(fig2)
    
    with col2:
        st.subheader("📈 Performance Metrics")
        metrics = calculate_metrics(df)
        
        for metric, value in metrics.items():
            st.metric(metric, value)
    
    # Data table
    with st.expander("📋 View Raw Data"):
        st.dataframe(
            df[['Close', f'MA_{fast_ma}', f'MA_{slow_ma}', 'Signal', 
                'Asset_Returns', 'Strategy_Returns']].round(4)
        )
    
    # Strategy explanation
    with st.expander("ℹ️ Strategy Explanation"):
        st.markdown(f"""
        **Moving Average Crossover Strategy:**
        - **Fast MA**: {fast_ma} days
        - **Slow MA**: {slow_ma} days
        - **Signal Generation**: Buy when fast MA > slow MA and volume > 20-day average
        - **Risk Management**: Only trade during high volume periods
        - **Position**: Long (+1), Short (-1), or Neutral (0)
        """)

if __name__ == "__main__":
    main()
