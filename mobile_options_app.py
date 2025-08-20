import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta, date
import os
from ta.trend import EMAIndicator
from nsepython import fnolist
import requests
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter
import mplfinance as mpf
import matplotlib.dates
import pytz

# For production:
from dhanhq import dhanhq
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure for mobile
st.set_page_config(
    page_title="Options Premium Tracker",
    page_icon="📱",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Mobile CSS
st.markdown("""
<style>
    .main > div { padding-top: 1rem; }
    .stButton > button { 
        width: 100%; 
        height: 3rem; 
        font-size: 1.2rem;
        background: linear-gradient(90deg, #1f4e79, #2e86ab);
        color: white;
        border: none;
        border-radius: 8px;
    }
    .metric-card {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #2e86ab;
        margin: 0.5rem 0;
    }
    @media (max-width: 768px) {
        .main .block-container { padding-left: 1rem; padding-right: 1rem; }
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'use_demo_data' not in st.session_state:
    st.session_state.use_demo_data = True

@st.cache_data
def get_fno_stocks():
    """Get list of F&O stocks from NSE"""
    try:
        fno_stocks = fnolist()
        # Sort alphabetically for better user experience
        return sorted(fno_stocks)
    except Exception as e:
        st.error(f"Error fetching F&O list: {e}")
        # Fallback to common stocks if API fails
        return ["TATAMOTORS", "RELIANCE", "HDFCBANK", "ICICIBANK", "SBIN", "TCS", "INFY", "ITC", "HINDUNILVR", "KOTAKBANK"]

def get_index_security_id(ticker):
    """Get hardcoded security IDs for indices from notebook"""
    index_security_ids = {
        "NIFTY": 13,
        "BANKNIFTY": 25,
        "FINNIFTY": 27,
        "MIDCPNIFTY": 442
    }
    return index_security_ids.get(ticker)

def fetch_ticker_secid(ticker):
    """Get security ID for a ticker from Dhan master file"""
    try:
        # Define index list
        INDEX_SYMBOLS = ["NIFTY", "BANKNIFTY", "FINNIFTY", "MIDCPNIFTY"]
        
        # For indices, return hardcoded security ID
        if ticker in INDEX_SYMBOLS:
            return get_index_security_id(ticker)

        # For stocks, use existing logic
        exchange, segment = "NSE", "EQUITY"
        all_mstr = pd.read_csv("https://images.dhan.co/api-data/api-scrip-master.csv")
        filter_df = all_mstr[
            (all_mstr["SEM_TRADING_SYMBOL"] == ticker) &
            (all_mstr["SEM_EXM_EXCH_ID"] == exchange) &
            (all_mstr["SEM_INSTRUMENT_NAME"] == segment)
        ]
        if filter_df.shape[0] == 1:
            return int(filter_df.iloc[0]["SEM_SMST_SECURITY_ID"])
        else:
            return None
    except Exception as e:
        st.error(f"Error fetching security ID for {ticker}: {e}")
        return None

def get_next_expiry(ticker):
    """Get the next Thursday expiry for options"""
    try:
        # Read from Kite instruments file
        all_mstr_expiry = pd.read_csv("https://api.kite.trade/instruments")
        
        findExpirydf = all_mstr_expiry[
            (all_mstr_expiry.exchange == "NFO") &
            (all_mstr_expiry.name == ticker) &
            (all_mstr_expiry.segment == "NFO-OPT")  # Changed from instrument_type == "FUT"
        ].reset_index(drop=True)
        
        if findExpirydf.empty:
            return None
            
        findExpirydf["expiry"] = pd.to_datetime(findExpirydf["expiry"]).apply(lambda x: x.date())
        findExpirydf.drop(findExpirydf[findExpirydf.expiry < datetime.now().date()].index, inplace=True)
        
        if findExpirydf.empty:
            return None
            
        weekly_expiry = findExpirydf["expiry"].unique().tolist()
        weekly_expiry.sort()
        return weekly_expiry[0].strftime("%Y-%m-%d")
        
    except Exception as e:
        st.error(f"Error getting expiry for {ticker}: {e}")
        return None

@st.cache_data(ttl=300)  # Cache for 5 minutes
def get_real_option_chain_data(ticker, _dhan_client):
    """Fetch real option chain data from Dhan API"""
    try:
        if not _dhan_client:
            return None
            
        # Determine if index or stock
        INDEX_SYMBOLS = ["NIFTY", "BANKNIFTY", "FINNIFTY", "MIDCPNIFTY"]
        is_index = ticker in INDEX_SYMBOLS
            
        # Get security ID
        secid = fetch_ticker_secid(ticker)
        if not secid:
            st.error(f"❌ Could not find security ID for {ticker}")
            return None
            
        # Get next expiry
        expiry = get_next_expiry(ticker)
        if not expiry:
            st.error(f"❌ Could not find expiry for {ticker}")
            return None
            
        # Get API credentials
        access_token = st.secrets.get("DHAN_API") if hasattr(st, 'secrets') else os.getenv("DHAN_API")
        client_id = st.secrets.get("DHAN_CLIENTID") if hasattr(st, 'secrets') else os.getenv("DHAN_CLIENTID")
        
        if not access_token or not client_id:
            st.error("❌ API credentials missing")
            return None
            
        # Dhan option chain API call
        url = "https://api.dhan.co/v2/optionchain"
        headers = {
            "access-token": access_token,
            "client-id": client_id,
            "Content-Type": "application/json",
        }
        
        payload = {
            "UnderlyingScrip": secid,
            "UnderlyingSeg": "IDX_I" if is_index else "NSE_FNO",  # Use IDX_I for indices
            "Expiry": expiry,
        }
        
        response = requests.post(url, headers=headers, json=payload)
        
        if response.status_code == 200:
            data = response.json()
            
            if "data" not in data or "oc" not in data["data"]:
                st.error(f"❌ Invalid option chain response for {ticker}")
                return None
                
            current_price = data["data"]["last_price"]
            option_chain = data["data"]["oc"]
            
            # Extract all available strikes from API (no filtering)
            strikes = []
            for strike_key in option_chain.keys():
                try:
                    strike_val = float(strike_key)
                    strikes.append(strike_val)
                except (ValueError, TypeError):
                    continue
            
            strikes.sort()
            
            if not strikes:
                st.warning(f"⚠️ API returned no valid strikes for {ticker}. Using demo data.")
                return None
            
            # Find ATM strike (closest available strike to current price)
            atm_strike = min(strikes, key=lambda x: abs(x - current_price))
            
            # Filter strikes to show only reasonable range around current price
            reasonable_strikes = [s for s in strikes if current_price * 0.7 <= s <= current_price * 1.3]
            if len(reasonable_strikes) < 5:  # If too few, expand the range
                reasonable_strikes = [s for s in strikes if current_price * 0.6 <= s <= current_price * 1.4]
            
            return {
                "current_price": current_price,
                "strikes": reasonable_strikes,  # Use filtered strikes
                "atm_strike": atm_strike,
                "expiry": expiry,
                "option_chain": option_chain,
            }
            
        else:
            st.error(f"❌ API Error {response.status_code}: {response.text}")
            return None
            
    except Exception as e:
        st.error(f"❌ Error fetching real option chain for {ticker}: {str(e)}")
        return None

@st.cache_data(ttl=60)  # Cache for 1 minute (shorter cache for quicker updates)
def get_option_chain_strikes(ticker, use_real_data=True):
    """Get available option strikes and current price for a ticker"""
    try:
        if use_real_data:
            # Initialize Dhan client
            dhan_client = init_dhan_client()
            if dhan_client:
                real_data = get_real_option_chain_data(ticker, dhan_client)
                if real_data and 'strikes' in real_data and len(real_data['strikes']) > 0:
                    # Validate that the strikes make sense for the ticker
                    current_price = real_data['current_price']
                    valid_strikes = [s for s in real_data['strikes'] if current_price * 0.6 <= s <= current_price * 1.4]
                    
                    if len(valid_strikes) >= 5:  # Need at least 5 valid strikes
                        real_data['strikes'] = valid_strikes
                        real_data['atm_strike'] = min(valid_strikes, key=lambda x: abs(x - current_price))
                        return real_data
        
        # If we couldn't get real data, show market closed message
        now = datetime.now()
        if now.weekday() >= 5:  # Weekend
            st.error("Market is closed. NSE operates Monday-Friday. Please try during market days.")
        else:
            st.error("Data is not available unless market opens. Please try this app during market hours (9:15 AM - 11:59 PM IST) on market days.")
        return None
        
    except Exception as e:
        st.error(f"Error fetching option chain for {ticker}: {e}")
        return None

def get_nearby_strikes(strikes, atm_strike, count=10):
    """Get strikes around ATM (±count strikes)"""
    try:
        strikes_sorted = sorted(strikes)
        atm_index = strikes_sorted.index(atm_strike)
        
        start_idx = max(0, atm_index - count)
        end_idx = min(len(strikes_sorted), atm_index + count + 1)
        
        return strikes_sorted[start_idx:end_idx]
    except:
        return strikes[:21]  # Return first 21 if error

def init_dhan_client():
    """Initialize Dhan client for production"""
    try:
        dhan_api = None
        client_id = None
        
        # Try Streamlit secrets first
        try:
            if hasattr(st, 'secrets'):
                dhan_api = st.secrets.get("DHAN_API")
                client_id = st.secrets.get("DHAN_CLIENTID")
        except Exception:
            pass
        
        # Fallback to environment variables
        if not dhan_api:
            dhan_api = os.getenv("DHAN_API")
        if not client_id:
            client_id = os.getenv("DHAN_CLIENTID")
        
        # Check if credentials are placeholder values
        if dhan_api and dhan_api != "your_dhan_api_key_here" and client_id and client_id != "your_dhan_client_id_here":
            return dhanhq(client_id, dhan_api)
        else:
            return None
    except Exception as e:
        st.error(f"Failed to connect to Dhan API: {e}")
        return None

def get_real_options_data(ticker, ltp, agg_tick, _dhan_client):
    """Fetch real options data using exact notebook logic"""
    try:
        if not _dhan_client:
            st.error("❌ Dhan client not initialized")
            return None
            
        # Determine if index or stock
        INDEX_SYMBOLS = ["NIFTY", "BANKNIFTY", "FINNIFTY", "MIDCPNIFTY"]
        is_index = ticker in INDEX_SYMBOLS
            
        # EXACT notebook logic - use Kite instruments file
        all_mstr_expiry = pd.read_csv("https://api.kite.trade/instruments")
        
        findExpirydf = all_mstr_expiry[
            (all_mstr_expiry.exchange == "NFO")
            & (all_mstr_expiry.name == ticker)
            & (all_mstr_expiry.segment == "NFO-OPT")
        ]
        
        # Find CE and PE instruments for the strike
        if findExpirydf.empty:
            st.error(f"❌ No options data found for {ticker} in NFO segment")
            return None
        
        # Get available strikes around the requested strike
        nearest_expiry = findExpirydf["expiry"].min()
        available_strikes = sorted(findExpirydf[findExpirydf["expiry"] == nearest_expiry]["strike"].unique())
        
        # Use wider range for indices
        strike_range = 500 if is_index else 200
        nearby_strikes = [s for s in available_strikes if abs(s - ltp) <= strike_range]
        
        instruments = findExpirydf[
            (findExpirydf["expiry"] == nearest_expiry)
            & (findExpirydf["strike"] == ltp)
        ]["exchange_token"].values.tolist()
        
        if len(instruments) != 2:
            # Try to find the closest available strike
            if nearby_strikes:
                closest_strike = min(nearby_strikes, key=lambda x: abs(x - ltp))
                instruments = findExpirydf[
                    (findExpirydf["expiry"] == nearest_expiry)
                    & (findExpirydf["strike"] == closest_strike)
                ]["exchange_token"].values.tolist()
                
                if len(instruments) == 2:
                    ltp = closest_strike  # Update the strike price
                else:
                    st.error(f"❌ Could not find both CE and PE instruments for {ticker}")
                    return None
            else:
                st.error(f"❌ No nearby strikes found for {ticker}")
                return None
        
        # Get current and previous trading day (same as notebook)
        today = datetime.now().date()
        yesterday = today - timedelta(days=1)
        while yesterday.weekday() >= 5:
            yesterday -= timedelta(days=1)
            
        from_date = yesterday.strftime("%Y-%m-%d")
        to_date = today.strftime("%Y-%m-%d")
        
        # Fetch data for both instruments (exact notebook logic)
        df_ce = None
        df_pe = None
        
        for secid in instruments:
            # Get intraday data - try with date range first, fallback to current day
            try:
                livefeed = _dhan_client.intraday_minute_data(
                    security_id=str(secid),
                    exchange_segment="NSE_FNO",
                    instrument_type="OPTIDX" if is_index else "OPTSTK",
                    from_date=from_date,
                    to_date=to_date
                )
            except TypeError as e:
                if "unexpected keyword argument" in str(e):
                    # Fallback to API without date parameters (gets current day data)
                    livefeed = _dhan_client.intraday_minute_data(
                        security_id=str(secid),
                        exchange_segment="NSE_FNO",
                        instrument_type="OPTIDX" if is_index else "OPTSTK"
                    )
                else:
                    raise e
            
            if not livefeed.get("data"):
                continue
                
            # Process data
            df = pd.DataFrame(livefeed["data"])
            
            # Handle timestamp conversion - try different field names
            timestamp_field = None
            if "start_Time" in df.columns:
                timestamp_field = "start_Time"
            elif "timestamp" in df.columns:
                timestamp_field = "timestamp"
            else:
                st.error(f"❌ No timestamp field found. Available columns: {df.columns.tolist()}")
                continue
            
            # Convert timestamps
            temp_list = []
            for i in df[timestamp_field]:
                temp = _dhan_client.convert_to_date_time(i)
                temp_list.append(temp)
            df["Timestamp"] = temp_list
            df["Timestamp"] = pd.to_datetime(df["Timestamp"])
            
            # Rename columns to match expected format
            df = df.rename(columns={
                'open': 'open',
                'high': 'high',
                'low': 'low',
                'close': 'close',
                'volume': 'volume' if 'volume' in df.columns else 'vol'
            })
            
            # Ensure all required columns exist with numeric values
            for col in ['open', 'high', 'low', 'close']:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
                else:
                    st.error(f"❌ Missing required column: {col}")
                    continue
            
            # Handle volume separately
            if 'volume' not in df.columns and 'vol' in df.columns:
                df['volume'] = df['vol']
            if 'volume' not in df.columns:
                df['volume'] = 0
            df['volume'] = pd.to_numeric(df['volume'], errors='coerce').fillna(0)
            
            # Determine if CE or PE based on exchange_token
            instrument_info = findExpirydf[findExpirydf["exchange_token"] == secid].iloc[0]
            if "CE" in instrument_info["tradingsymbol"]:
                df_ce = df.copy()
            elif "PE" in instrument_info["tradingsymbol"]:
                df_pe = df.copy()
        
        if df_ce is None or df_pe is None:
            st.error(f"❌ Could not get both CE and PE data for {ticker}")
            return None
        
        # EXACT notebook combination logic
        df_combined = (
            df_ce[["Timestamp", "open", "high", "low", "close", "volume"]]
            .rename(columns={
                "open": "ce_open", "high": "ce_high", "low": "ce_low",
                "close": "ce_close", "volume": "ce_volume"
            })
            .merge(
                df_pe[["Timestamp", "open", "high", "low", "close", "volume"]].rename(columns={
                    "open": "pe_open", "high": "pe_high", "low": "pe_low",
                    "close": "pe_close", "volume": "pe_volume"
                }),
                on="Timestamp"
            )
        )
        
        # CRITICAL: Calculate prev close from previous day data, then filter to current day only
        prev_day_premium_close = None
        if df_combined[df_combined["Timestamp"] < to_date].shape[0] > 0:
            prev_day_premium_close = float(
                df_combined[df_combined["Timestamp"] < to_date]
                .tail(1)[["ce_close", "pe_close"]]
                .sum(axis=1)
                .iloc[0]
            )
        
        # EXACT notebook logic: Filter to current day data only (>=to_date)
        df_combined = df_combined[df_combined["Timestamp"] >= to_date]
        
        if df_combined.empty:
            st.error(f"❌ No data available for current day for {ticker}")
            return None
        
        # Calculate combined straddle values (exact notebook logic)
        df_combined["combined_open"] = df_combined["ce_open"] + df_combined["pe_open"]
        df_combined["combined_close"] = df_combined["ce_close"] + df_combined["pe_close"]
        df_combined["combined_high"] = np.maximum(
            df_combined["ce_high"] + df_combined["pe_low"],
            df_combined["ce_low"] + df_combined["pe_high"]
        )
        df_combined["combined_low"] = np.minimum(
            df_combined["ce_low"] + df_combined["pe_high"],
            df_combined["ce_high"] + df_combined["pe_low"]
        )
        
        # Ensure logical consistency (exact notebook logic)
        df_combined["combined_high"] = np.maximum(
            df_combined["combined_high"],
            np.maximum(df_combined["combined_open"], df_combined["combined_close"])
        )
        df_combined["combined_low"] = np.minimum(
            df_combined["combined_low"],
            np.minimum(df_combined["combined_open"], df_combined["combined_close"])
        )
        
        # Handle volume data - ensure it exists and is numeric
        df_combined["ce_volume"] = pd.to_numeric(df_ce["volume"], errors='coerce').fillna(0)
        df_combined["pe_volume"] = pd.to_numeric(df_pe["volume"], errors='coerce').fillna(0)
        df_combined["combined_volume"] = df_combined["ce_volume"] + df_combined["pe_volume"]
        df_combined.set_index("Timestamp", inplace=True)
        
        # Resample (exact notebook logic)
        df_resampled = df_combined.resample(agg_tick).agg({
            "combined_open": "first",
            "combined_high": "max", 
            "combined_low": "min",
            "combined_close": "last",
            "combined_volume": "sum",
            "ce_close": "last",
            "pe_close": "last"
        }).dropna()
        
        # Add EMAs (exact notebook logic)
        df_resampled["ema9"] = EMAIndicator(close=df_resampled["combined_close"], window=9).ema_indicator()
        df_resampled["ema21"] = EMAIndicator(close=df_resampled["combined_close"], window=15).ema_indicator()
        
        df_resampled["strike"] = ltp
        df_resampled["is_real_data"] = True
        
        # Add previous day close for chart reference
        if prev_day_premium_close is not None:
            df_resampled["prev_day_close"] = prev_day_premium_close
        
        return df_resampled
        
    except Exception as e:
        st.error(f"❌ Error fetching real options data: {str(e)}")
        return None

def create_advanced_chart(df, ticker, ltp, show_prev_close, show_emas=True):
    """Create production-quality chart"""
    
    fig = go.Figure()
    
    # Candlestick
    fig.add_trace(go.Candlestick(
        x=df.index,
        open=df['combined_open'],
        high=df['combined_high'],
        low=df['combined_low'],
        close=df['combined_close'],
        name="Straddle Premium",
        increasing_fillcolor='#26a69a',
        increasing_line_color='#26a69a',
        decreasing_fillcolor='#ef5350',
        decreasing_line_color='#ef5350'
    ))
    
    # EMAs
    if show_emas:
        fig.add_trace(go.Scatter(
            x=df.index,
            y=df['ema9'],
            mode='lines',
            name='EMA 9',
            line=dict(color='#2196f3', width=2)
        ))
        
        fig.add_trace(go.Scatter(
            x=df.index,
            y=df['ema21'],
            mode='lines',
            name='EMA 21',
            line=dict(color='#ff9800', width=2)
        ))
    
    # Previous close - use actual previous day close if available
    if show_prev_close:
        if 'prev_day_close' in df.columns and not pd.isna(df['prev_day_close'].iloc[0]):
            prev_close = df['prev_day_close'].iloc[0]
        else:
            # Fallback for demo data
            prev_close = df['combined_close'].iloc[0] * 0.98
            
        fig.add_hline(
            y=prev_close,
            line_dash="dash",
            line_color="gray",
            annotation_text=f"Prev Close: ₹{prev_close:.1f}",
            annotation_position="top right"
        )
    
    # Get moneyness info for title
    moneyness_text = ""
    if 'moneyness' in df.columns and not df['moneyness'].empty:
        moneyness = df['moneyness'].iloc[-1]
        if moneyness > 0.05:
            moneyness_text = " (OTM Call)"
        elif moneyness < -0.05:
            moneyness_text = " (OTM Put)"
        else:
            moneyness_text = " (ATM)"
    
    # Mobile-optimized layout
    fig.update_layout(
        title={
            'text': f"{ticker} ₹{ltp} Straddle{moneyness_text}",
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 20, 'color': '#1f4e79'}
        },
        height=450,
        margin=dict(l=10, r=10, t=60, b=40),
        xaxis_rangeslider_visible=False,
        template="plotly_white",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        xaxis=dict(title="Time", title_font_size=14),
        yaxis=dict(title="Premium (₹)", title_font_size=14),
        hovermode='x unified'
    )
    
    return fig

def create_static_chart(df, ticker, ltp, show_prev_close, show_emas=True):
    """Create static matplotlib chart that mirrors the original Plotly version"""
    # Debug print to see timestamps
    print("Timestamps in data:", df.index[0], "to", df.index[-1])
    
    # Create figure and axis
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Convert timestamps while preserving timezone
    dates = df.index.map(lambda x: matplotlib.dates.date2num(x.tz_localize(None)))
    
    # Plot candlesticks
    width = 0.8 * (dates[1] - dates[0]) if len(dates) > 1 else 0.0002  # Dynamic width based on time difference
    
    up = df[df['combined_close'] >= df['combined_open']]
    down = df[df['combined_close'] < df['combined_open']]
    
    # Plot up candlesticks
    for i, idx in enumerate(up.index):
        date = dates[df.index.get_loc(idx)]
        # Body
        body_bottom = min(up.loc[idx, 'combined_open'], up.loc[idx, 'combined_close'])
        body_height = abs(up.loc[idx, 'combined_close'] - up.loc[idx, 'combined_open'])
        rect = plt.Rectangle((date - width/2, body_bottom), width, body_height,
                           fill=True, color='#26a69a')
        ax.add_patch(rect)
        # Wick
        ax.plot([date, date], 
                [up.loc[idx, 'combined_low'], up.loc[idx, 'combined_high']], 
                color='#26a69a', linewidth=1)
    
    # Plot down candlesticks
    for i, idx in enumerate(down.index):
        date = dates[df.index.get_loc(idx)]
        # Body
        body_bottom = min(down.loc[idx, 'combined_open'], down.loc[idx, 'combined_close'])
        body_height = abs(down.loc[idx, 'combined_close'] - down.loc[idx, 'combined_open'])  # Fixed height calculation
        rect = plt.Rectangle((date - width/2, body_bottom), width, body_height,
                           fill=True, color='#ef5350')
        ax.add_patch(rect)
        # Wick
        ax.plot([date, date], 
                [down.loc[idx, 'combined_low'], down.loc[idx, 'combined_high']], 
                color='#ef5350', linewidth=1)
    
    # EMAs
    if show_emas:
        ax.plot(dates, df['ema9'], label='EMA 9', color='#2196f3', linewidth=2)
        ax.plot(dates, df['ema21'], label='EMA 21', color='#ff9800', linewidth=2)
        ax.legend()
    
    # Previous close line
    if show_prev_close and 'prev_day_close' in df.columns:
        prev_close = df['prev_day_close'].iloc[0]
        if not pd.isna(prev_close):
            ax.axhline(y=prev_close, color='gray', linestyle='--', alpha=0.5)
            ax.text(dates[-1], prev_close, f'Prev Close: ₹{prev_close:.1f}', 
                    verticalalignment='bottom', horizontalalignment='right')
    
    # Customize the plot
    plt.title(f"{ticker} ₹{ltp} Straddle", pad=20, fontsize=14)
    plt.xlabel('Time')
    plt.ylabel('Premium (₹)')
    
    # Format x-axis
    ax.xaxis.set_major_formatter(DateFormatter('%H:%M'))
    plt.xticks(rotation=45)
    
    # Set y-axis limits with some padding
    ymin = df['combined_low'].min() * 0.995
    ymax = df['combined_high'].max() * 1.005
    plt.ylim(ymin, ymax)
    
    # Grid
    ax.grid(True, alpha=0.2)
    
    # Adjust layout
    plt.tight_layout()
    
    return fig

# Main App
st.title("📱 Stock-Options Straddle App")
st.markdown("*Web App for tracking straddle premiums for FnO stocks on NSE*")

# Sidebar for settings
with st.sidebar:
    st.header("⚙️ Settings")
    use_real_data = st.checkbox("Use Real API Data", value=False, help="Toggle between demo and real Dhan API data")
    show_emas = st.checkbox("Show EMA Lines", value=True)
    auto_refresh = st.selectbox("Auto Refresh", ["Off", "30s", "1min", "5min"])
    
    st.markdown("---")
    st.markdown("**📊 About**")
    st.markdown("This app tracks options straddle premiums in real-time, perfect for mobile trading.")
    
    # Show F&O stocks count
    try:
        fno_count = len(get_fno_stocks())
        st.markdown(f"**{fno_count}** F&O stocks available")
    except:
        pass

# Get F&O stocks list and add indices
INDEX_SYMBOLS = ["NIFTY", "BANKNIFTY", "FINNIFTY", "MIDCPNIFTY"]
fno_stocks = get_fno_stocks()

# Add indices at the beginning of the list
all_instruments = INDEX_SYMBOLS + fno_stocks
filtered_stocks = all_instruments

# Ticker selection
ticker_col1, ticker_col2 = st.columns([2, 1])

with ticker_col1:
    # Find default index for TATAMOTORS
    default_index = 0
    if "TATAMOTORS" in filtered_stocks:
        default_index = filtered_stocks.index("TATAMOTORS")
    elif len(filtered_stocks) > 0:
        default_index = 0
    
    ticker = st.selectbox(
        "📈 Stock Symbol", 
        options=filtered_stocks if filtered_stocks else fno_stocks,
        index=default_index,
        help="Select NSE F&O stock symbol",
        key="ticker_select"
    )

# Get option chain data for selected ticker (updates immediately when ticker changes)
use_real_data = True
option_data = get_option_chain_strikes(ticker, use_real_data)

# Store in session state to avoid stale data issues
st.session_state['current_ticker'] = ticker
st.session_state['current_option_data'] = option_data

if option_data:
    current_price = option_data["current_price"]
    atm_strike = option_data["atm_strike"]
    available_strikes = get_nearby_strikes(option_data["strikes"], atm_strike, 10)
    
    # Show data source and current price info with refresh button
    is_demo = option_data.get("is_demo", False)
    data_source = "🔴 Demo Data" if is_demo else "🟢 Live Data"
    expiry_info = f" | Expiry: {option_data.get('expiry', 'N/A')}" if not is_demo else ""
    
    # Live data info with refresh button
    info_col1, info_col2 = st.columns([4, 1])
    with info_col1:
        if is_demo:
            st.warning(f"{data_source} | **{ticker}** Demo Price: ₹{current_price:.1f} | ATM: ₹{atm_strike}{expiry_info}")
        else:
            st.success(f"{data_source} | **{ticker}** Live Price: ₹{current_price:.1f} | ATM: ₹{atm_strike}{expiry_info}")
    with info_col2:
        if st.button("🔄", help="Refresh option data", key="refresh_options"):
            st.cache_data.clear()
            # Clear session state as well
            if 'current_ticker' in st.session_state:
                del st.session_state['current_ticker']
            if 'current_option_data' in st.session_state:
                del st.session_state['current_option_data']
            st.rerun()
    

else:
    st.error("❌ Could not fetch option data")
    available_strikes = [650.0]  # Fallback
    atm_strike = 650.0
    current_price = 650.0

# Input form for other parameters
with st.form("trading_params", clear_on_submit=False):
    col1, col2 = st.columns([1, 1])
    
    with col1:
        agg_tick = st.selectbox(
            "📊 Timeframe",
            ["1T", "3T", "5T", "15T"],
            index=1,
            help="Chart aggregation interval"
        )
    
    with col2:
        if 'available_strikes' in locals() and available_strikes:
            # Strike price dropdown with valid options
            try:
                atm_index = available_strikes.index(atm_strike)
            except:
                atm_index = len(available_strikes) // 2
                
            ltp = st.selectbox(
                "🎯 Strike Price",
                options=available_strikes,
                index=atm_index,
                help="Select from available option strikes (ATM ± 10 strikes)"
            )
        else:
            ltp = st.number_input(
                "💰 Strike Price",
                value=650.0,
                min_value=1.0,
                step=1.0,
                help="Manual strike price (option data unavailable)"
            )
        
        show_prev_close = st.checkbox(
            "📉 Previous Close Line",
            value=True
        )
    
    # Generate button
    submitted = st.form_submit_button(
        "🚀 Generate Live Chart",
        use_container_width=True
    )

# Main logic
if submitted:
    with st.spinner("🔄 Fetching options data..."):
        
        # Get current ticker and option data from session state
        current_ticker = st.session_state.get('current_ticker', ticker)
        current_option_data = st.session_state.get('current_option_data', option_data)
        
        # Initialize client
        dhan_client = init_dhan_client()
        
        try:
            # Always try to get real data first
            if dhan_client:
                df_data = get_real_options_data(current_ticker, ltp, agg_tick, dhan_client)
                if df_data is None:
                    st.error("❌ Could not fetch data. Please check your inputs and try again.")
                    st.stop()
            else:
                st.error("❌ Dhan client not initialized. Please check your API credentials.")
                st.stop()
            
            # Create static chart instead of plotly chart
            fig = create_static_chart(df_data, current_ticker, ltp, show_prev_close, show_emas)
            st.pyplot(fig)
            
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")
            st.error("Please check your inputs and try again.")



# Footer
st.markdown("""
<div style='text-align: center; color: #666; font-size: 0.9em; margin: 2rem 0;'>
🛠️ Built with Streamlit | 📊 Real-time Options Tracking | 📱 An App by Pappupedia
</div>
""", unsafe_allow_html=True) 