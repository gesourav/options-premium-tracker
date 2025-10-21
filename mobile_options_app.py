#!/usr/bin/env python3
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

import sourcedefender
from dhan_token_automate import GetAccessToken

# Load environment variables
load_dotenv()

# === ACCESS TOKEN CACHING (MOVED TO MODULE LEVEL) ===
@st.cache_data(ttl=300*60)  # Cache for 5 hours
def get_cached_access_token():
    """Get and cache Dhan access token"""
    try:
        # Try Streamlit secrets first
        if hasattr(st, 'secrets'):
            mobile = st.secrets.get("DHAN_MOBILE_NO")
            client_id = st.secrets.get("DHAN_CLIENTID")
            api_key = st.secrets.get("DHAN_API_KEY")
            api_secret = st.secrets.get("DHAN_API_SECRET")
            totp_key = st.secrets.get("DHAN_TOTP_KEY")
            pin = st.secrets.get("DHAN_USER_PIN")
        else:
            # Fallback to environment variables
            mobile = os.getenv("DHAN_MOBILE_NO")
            client_id = os.getenv("DHAN_CLIENTID")
            api_key = os.getenv("DHAN_API_KEY")
            api_secret = os.getenv("DHAN_API_SECRET")
            totp_key = os.getenv("DHAN_TOTP_KEY")
            pin = os.getenv("DHAN_USER_PIN")
        
        if all([mobile, client_id, api_key, api_secret, totp_key, pin]):
            return GetAccessToken(mobile, client_id, api_key, api_secret, totp_key, pin)
        return None
    except Exception as e:
        st.error(f"Error getting access token: {e}")
        return None

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
    /* Style for recommendation table */
    .recommendation-table {
        font-size: 0.9em;
    }
    .strong-sell { background-color: #d4edda !important; }
    .sell { background-color: #f8f9fa !important; }
    .consider { background-color: #fff3cd !important; }
    .weak { background-color: #f8d7da !important; }
    .avoid { background-color: #f5c6cb !important; }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'use_demo_data' not in st.session_state:
    st.session_state.use_demo_data = True

# === ENHANCED OPTION CHAIN FUNCTIONS ===

def calculate_extrinsic_value(option_price, strike, underlying_price, option_type):
    """Calculate extrinsic value of an option"""
    if pd.isna(option_price) or option_price == 0:
        return 0
    
    if option_type == 'CE':  # Call option
        intrinsic_value = max(0, underlying_price - strike)
    else:  # Put option (PE)
        intrinsic_value = max(0, strike - underlying_price)
    
    extrinsic_value = option_price - intrinsic_value
    return max(0, extrinsic_value)

def calculate_advanced_greek_ratios(options_df, underlying_price, ivp=50, dte=30):
    """
    Calculate advanced Greek ratios and trading metrics for options selection
    """
    
    # Make a copy to avoid modifying original
    enhanced_df = options_df.copy()
    
    # IVP Environment Classification
    if ivp >= 80:
        iv_environment = "VERY_HIGH"
        iv_factor = 1.5
        vega_tolerance = 1.2
    elif ivp >= 60:
        iv_environment = "HIGH" 
        iv_factor = 1.3
        vega_tolerance = 1.0
    elif ivp >= 40:
        iv_environment = "MODERATE"
        iv_factor = 1.1
        vega_tolerance = 0.8
    elif ivp >= 20:
        iv_environment = "LOW"
        iv_factor = 0.9
        vega_tolerance = 0.6
    else:
        iv_environment = "VERY_LOW"
        iv_factor = 0.7
        vega_tolerance = 0.4
    
    # Add extrinsic value calculation
    enhanced_df['extrinsic_value'] = enhanced_df.apply(
        lambda row: calculate_extrinsic_value(
            row['last_price'], row['strike'], underlying_price, row['type']
        ), axis=1
    ).round(2)
    
    # Calculate straddle prices by grouping strikes
    straddle_df = enhanced_df.groupby('strike').agg({
        'last_price': 'sum'
    }).rename(columns={'last_price': 'straddle_price'}).round(0)
    
    # Merge straddle prices back
    enhanced_df = enhanced_df.merge(straddle_df, on='strike', how='left')
    
    # === PRIMARY GREEK RATIOS ===
    
    # 1. Theta/Delta Ratio
    enhanced_df['theta_delta_ratio'] = np.where(
        enhanced_df['delta'].abs() > 0.01,
        enhanced_df['theta'].abs() / enhanced_df['delta'].abs(),
        np.nan
    ).round(2)
    
    # 2. IVP-Adjusted Theta/Vega Ratio 
    enhanced_df['theta_vega_ratio'] = np.where(
        enhanced_df['vega'] > 0.1,
        (enhanced_df['theta'].abs() / enhanced_df['vega']) * vega_tolerance,
        np.nan
    ).round(3)
    
    # 3. Gamma-Adjusted Delta Efficiency
    annual_vol = 0.20
    gamma_adjustment = enhanced_df['gamma'] * (underlying_price ** 2) * (annual_vol ** 2) / 365
    enhanced_df['gamma_adj_efficiency'] = np.where(
        enhanced_df['delta'].abs() > 0.01,
        (enhanced_df['theta'].abs() - gamma_adjustment) / enhanced_df['delta'].abs(),
        np.nan
    ).round(2)
    
    # 4. IVP-Weighted Greek Efficiency Score
    gamma_risk_factor = np.minimum(enhanced_df['gamma'] * 10, 0.5)
    
    enhanced_df['efficiency_score'] = (
        enhanced_df['theta_delta_ratio'] * 
        iv_factor * 
        enhanced_df['theta_vega_ratio'].fillna(1) * 
        (1 - gamma_risk_factor)
    ).round(2)
    
    # 5. OTM Selling Attractiveness Score
    def calculate_selling_score(row, ivp):
        score = 0
        
        # Delta range scoring
        abs_delta = abs(row['delta']) if not pd.isna(row['delta']) else 0
        
        if ivp >= 60:
            optimal_delta_range = (0.12, 0.28)
        else:
            optimal_delta_range = (0.15, 0.25)
        
        if optimal_delta_range[0] <= abs_delta <= optimal_delta_range[1]:
            score += 35
        elif optimal_delta_range[0] - 0.05 <= abs_delta <= optimal_delta_range[1] + 0.05:
            score += 25
        elif abs_delta < optimal_delta_range[0]:
            score += 15
        
        # Theta scoring
        if not pd.isna(row['theta']) and row['theta'] < 0:
            daily_theta = abs(row['theta'])
            theta_threshold = 2.5 if ivp >= 60 else 3.0
            
            if daily_theta > theta_threshold * 1.2:
                score += 30
            elif daily_theta > theta_threshold:
                score += 25
            elif daily_theta > theta_threshold * 0.7:
                score += 20
            else:
                score += 10
        
        # Vega scoring
        if not pd.isna(row['vega']):
            vega_threshold = 5 if ivp >= 60 else 3
            
            if row['vega'] < vega_threshold * 0.6:
                score += 25
            elif row['vega'] < vega_threshold:
                score += 20
            elif row['vega'] < vega_threshold * 1.5:
                score += 15
            else:
                score += 5
        
        # Gamma scoring
        if not pd.isna(row['gamma']):
            if row['gamma'] < 0.002:
                score += 10
            elif row['gamma'] < 0.005:
                score += 7
            else:
                score += 3
        
        return min(score, 100)
    
    enhanced_df['selling_score'] = enhanced_df.apply(
        lambda row: calculate_selling_score(row, ivp), axis=1
    )
    
    # 6. Expected Value Calculation
    def calculate_expected_value(row, dte, ivp):
        if pd.isna(row['last_price']) or pd.isna(row['theta']) or pd.isna(row['delta']) or row['last_price'] <= 0:
            return np.nan
        
        # Probability ITM
        prob_itm = abs(row['delta'])
        
        # IVP affects volatility expansion risk
        vol_expansion_risk = 1.0
        if ivp >= 80:
            vol_expansion_risk = 0.8
        elif ivp >= 60:
            vol_expansion_risk = 0.9
        elif ivp <= 20:
            vol_expansion_risk = 1.3
        elif ivp <= 40:
            vol_expansion_risk = 1.1
        
        # Expected theta collection
        theta_collection = abs(row['theta']) * dte * (1 - prob_itm) * vol_expansion_risk
        
        # Expected loss
        avg_loss_factor = 0.25 if ivp >= 60 else 0.35
        expected_loss = row['last_price'] * prob_itm * avg_loss_factor
        
        return theta_collection - expected_loss
    
    enhanced_df['expected_value'] = enhanced_df.apply(
        lambda row: calculate_expected_value(row, dte, ivp), axis=1
    ).round(2)
    
    # 7. Trade Recommendations with ITM Safety Filter
    def get_recommendation(row, ivp):
        if pd.isna(row['selling_score']) or pd.isna(row['efficiency_score']):
            return "NO_DATA"
        
        # CRITICAL SAFETY CHECK: Never recommend selling ITM options
        # ITM means abs(delta) > 0.5, which carries very high assignment risk
        if pd.notna(row['delta']) and abs(row['delta']) > 0.5:
            return "AVOID"  # ITM options are too risky to sell
        
        # Additional safety: Don't recommend if delta is too close to 0.5 (near ITM)
        if pd.notna(row['delta']) and abs(row['delta']) > 0.45:
            return "WEAK"  # Close to ITM, high risk
        
        # Adjust thresholds based on IVP for OTM options only
        if ivp >= 70:
            strong_threshold, sell_threshold = 70, 55
        elif ivp >= 50:
            strong_threshold, sell_threshold = 75, 60
        else:
            strong_threshold, sell_threshold = 80, 65
        
        if row['selling_score'] >= strong_threshold and row['efficiency_score'] >= 2.5:
            return "STRONG_SELL"
        elif row['selling_score'] >= sell_threshold and row['efficiency_score'] >= 1.8:
            return "SELL"
        elif row['selling_score'] >= 45 and row['efficiency_score'] >= 1.2:
            return "CONSIDER"
        elif row['selling_score'] >= 30:
            return "WEAK"
        else:
            return "AVOID"
    
    enhanced_df['recommendation'] = enhanced_df.apply(
        lambda row: get_recommendation(row, ivp), axis=1
    )
    
    # Add metadata
    enhanced_df['ivp_environment'] = iv_environment
    enhanced_df['ivp_value'] = ivp
    enhanced_df['dte'] = dte
    
    return enhanced_df

def process_option_chain_for_recommendations(option_chain_data, underlying_price, ivp=50, dte=30):
    """
    Process raw option chain data from Dhan API into enhanced recommendation format
    """
    try:
        flattened_data = []
        
        for strike, data in option_chain_data.items():
            for option_type, option_data in data.items():
                greeks = option_data.pop('greeks', {})
                row = {
                    'strike': float(strike),
                    'type': option_type.upper(),
                    **greeks,
                    **option_data
                }
                flattened_data.append(row)
        
        # Create DataFrame
        df = pd.DataFrame(flattened_data)
        
        # Rename columns to match our enhanced functions
        df = df.rename(columns={
            'underlying_last_price': 'underlying_price'
        })
        
        # Add underlying price if not present
        if 'underlying_price' not in df.columns:
            df['underlying_price'] = underlying_price
        
        # Filter for strikes around ATM (±10 strikes only for better focus)
        strikes = sorted(df['strike'].unique())
        atm_strike = min(strikes, key=lambda x: abs(x - underlying_price))
        
        try:
            atm_index = strikes.index(atm_strike)
            start_idx = max(0, atm_index - 10)
            end_idx = min(len(strikes), atm_index + 11)  # +11 to include ATM
            selected_strikes = strikes[start_idx:end_idx]
            reasonable_strikes = df[df['strike'].isin(selected_strikes)].copy()
        except:
            # Fallback to original method if ATM index lookup fails
            reasonable_strikes = df[
                (df['strike'] >= underlying_price * 0.9) & 
                (df['strike'] <= underlying_price * 1.1)
            ].copy()
        
        if reasonable_strikes.empty:
            return None
        
        # Calculate enhanced metrics
        enhanced_df = calculate_advanced_greek_ratios(reasonable_strikes, underlying_price, ivp, dte)
        
        return enhanced_df
        
    except Exception as e:
        st.error(f"Error processing option chain data: {e}")
        return None

def display_recommendations_table(enhanced_df, underlying_price):
    """Display enhanced recommendations table"""
    
    if enhanced_df is None or enhanced_df.empty:
        st.warning("No option chain data available for recommendations.")
        return
    
    st.subheader("📊 Option Selling Recommendations")
    
    # Create separate tables for calls and puts
    calls_df = enhanced_df[enhanced_df['type'] == 'CE'].copy()
    puts_df = enhanced_df[enhanced_df['type'] == 'PE'].copy()
    
    # Select key columns for display
    display_columns = [
        'strike', 'last_price', 'delta', 'theta', 'vega', 'gamma', 
        'implied_volatility', 'extrinsic_value', 'straddle_price',
        'theta_delta_ratio', 'theta_vega_ratio', 'efficiency_score', 
        'selling_score', 'expected_value', 'recommendation'
    ]
    
    # Format the data for better display
    def format_display_df(df):
        display_df = df[display_columns].copy()
        
        # Round and format columns for better readability and space efficiency
        display_df['strike'] = display_df['strike'].astype(int)
        display_df['last_price'] = display_df['last_price'].round(2)
        display_df['delta'] = display_df['delta'].round(2)  # Reduced from 3 to 2 decimals
        display_df['theta'] = display_df['theta'].round(1)
        display_df['vega'] = display_df['vega'].round(1)
        display_df['gamma'] = display_df['gamma'].round(3)  # Reduced from 4 to 3 decimals
        display_df['implied_volatility'] = (display_df['implied_volatility']).round(2)
        display_df['extrinsic_value'] = display_df['extrinsic_value'].round(2)
        display_df['straddle_price'] = display_df['straddle_price'].astype(int)
        display_df['theta_delta_ratio'] = display_df['theta_delta_ratio'].round(1)
        display_df['theta_vega_ratio'] = display_df['theta_vega_ratio'].round(1)
        display_df['efficiency_score'] = display_df['efficiency_score'].round(1)
        display_df['selling_score'] = display_df['selling_score'].round(0).astype(int)
        display_df['expected_value'] = display_df['expected_value'].round(1)

        # Convert to string with proper formatting to avoid scientific notation
        for col in ['last_price', 'delta', 'theta', 'vega', 'implied_volatility', 'extrinsic_value', 'theta_delta_ratio', 
                'theta_vega_ratio', 'efficiency_score', 'expected_value']:
            if col in display_df.columns:
                display_df[col] = display_df[col].apply(lambda x: f"{x:.2f}" if pd.notna(x) else "0.00")

        for col in ['gamma']:
            if col in display_df.columns:
                display_df[col] = display_df[col].apply(lambda x: f"{x:.3f}" if pd.notna(x) else "0.000")
        
        # Rename columns for better readability with shorter names for mobile
        display_df = display_df.rename(columns={
            'strike': 'Strike',
            'last_price': 'Price',
            'delta': 'Δ',
            'theta': 'Θ',
            'vega': 'ν', 
            'gamma': 'Γ',
            'implied_volatility': 'IV',
            'extrinsic_value': 'Extr',
            'straddle_price': 'Strd',
            'theta_delta_ratio': 'Θ/Δ',
            'theta_vega_ratio': 'Θ/ν',
            'efficiency_score': 'Eff',
            'selling_score': 'Score',
            'expected_value': 'ExpV',
            'recommendation': 'Rec'
        })
        
        return display_df
    
    # Display tables in columns
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**📈 CALL OPTIONS**")
        if not calls_df.empty:
            calls_display = format_display_df(calls_df)
            
            # Style the dataframe based on recommendations with better contrast
            def style_recommendations(row):
                rec = row['Rec']
                if rec == 'STRONG_SELL':
                    return ['background-color: #d4edda; color: #155724; font-weight: bold'] * len(row)
                elif rec == 'SELL':
                    return ['background-color: #cce5ff; color: #004085; font-weight: bold'] * len(row)
                elif rec == 'CONSIDER':
                    return ['background-color: #fff3cd; color: #856404; font-weight: bold'] * len(row)
                elif rec == 'WEAK':
                    return ['background-color: #f8d7da; color: #721c24; font-weight: bold'] * len(row)
                else:
                    return ['background-color: #f5c6cb; color: #721c24; font-weight: bold'] * len(row)
            
            styled_calls = calls_display.style.apply(style_recommendations, axis=1)
            st.dataframe(styled_calls, use_container_width=True, height=400)
        else:
            st.info("No call options data available")
    
    with col2:
        st.markdown("**📉 PUT OPTIONS**")
        if not puts_df.empty:
            puts_display = format_display_df(puts_df)
            
            # Use the same styling function for puts
            def style_recommendations_puts(row):
                rec = row['Rec']
                if rec == 'STRONG_SELL':
                    return ['background-color: #d4edda; color: #155724; font-weight: bold'] * len(row)
                elif rec == 'SELL':
                    return ['background-color: #cce5ff; color: #004085; font-weight: bold'] * len(row)
                elif rec == 'CONSIDER':
                    return ['background-color: #fff3cd; color: #856404; font-weight: bold'] * len(row)
                elif rec == 'WEAK':
                    return ['background-color: #f8d7da; color: #721c24; font-weight: bold'] * len(row)
                else:
                    return ['background-color: #f5c6cb; color: #721c24; font-weight: bold'] * len(row)
            
            styled_puts = puts_display.style.apply(style_recommendations_puts, axis=1)
            st.dataframe(styled_puts, use_container_width=True, height=400)
        else:
            st.info("No put options data available")
    
    # Legend
    st.markdown("""
    **Legend:** 
    🟢 STRONG_SELL | 🔵 SELL | 🟡 CONSIDER | 🔴 WEAK | ⚫ AVOID
    
    **Key Metrics:**
    - **Θ/Δ**: Theta/Delta ratio (higher = better time decay efficiency)
    - **Θ/ν**: Theta/Vega ratio (higher = lower volatility risk)  
    - **Eff**: Composite Greek efficiency score
    - **Score**: Overall selling attractiveness (0-100)
    - **ExpV**: Risk-adjusted expected value
    - **Extr**: Extrinsic value (time value component)
    - **Strd**: Straddle price (Call + Put premium)
    
    **Note**: Showing ATM ± 10 strikes for optimal liquidity focus
    """)

# === EXISTING FUNCTIONS (UNCHANGED) ===

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
            
        # Get API credentials using cached function
        access_token = get_cached_access_token()
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
        # Get cached access token
        dhan_api = get_cached_access_token()
        
        # Get client ID
        client_id = None
        if hasattr(st, 'secrets'):
            client_id = st.secrets.get("DHAN_CLIENTID")
        else:
            client_id = os.getenv("DHAN_CLIENTID")
        
        # Check if credentials are valid
        if dhan_api and dhan_api != "your_dhan_api_key_here" and client_id and client_id != "your_dhan_client_id_here":
            return dhanhq(client_id, dhan_api)
        else:
            st.warning("⚠️ Dhan API credentials not configured properly")
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
                st.warning(f"⚠️ No intraday data returned for security ID {secid}")
                continue
            
            # Handle both list of dicts and dict of lists formats
            data_content = livefeed["data"]
            
            if isinstance(data_content, list):
                if len(data_content) == 0:
                    st.warning(f"⚠️ Empty data for security ID {secid}")
                    continue
                df = pd.DataFrame(data_content)
            elif isinstance(data_content, dict):
                # Columnar format (dict of lists) - common with Dhan API
                df = pd.DataFrame(data_content)
            else:
                st.warning(f"⚠️ Invalid data structure for security ID {secid}")
                continue
            
            # Validate DataFrame is not empty
            if df.empty:
                st.warning(f"⚠️ DataFrame is empty for security ID {secid}")
                continue
            
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
        
        # Validate dataframes are not empty
        if df_ce.empty or df_pe.empty:
            st.error(f"❌ One or both option dataframes are empty for {ticker}")
            return None
        
        # Validate required columns exist
        required_cols = ["Timestamp", "open", "high", "low", "close", "volume"]
        if not all(col in df_ce.columns for col in required_cols):
            st.error(f"❌ CE data missing required columns. Available: {df_ce.columns.tolist()}")
            return None
        if not all(col in df_pe.columns for col in required_cols):
            st.error(f"❌ PE data missing required columns. Available: {df_pe.columns.tolist()}")
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
st.title("📱 Options Selling App")
st.markdown("*Web App for tracking straddle premiums/Option Chain for FnO stocks / Major Indexes on NSE*")

# Sidebar for settings
with st.sidebar:
    st.header("⚙️ Settings")
    use_real_data = st.checkbox("Use Real API Data", value=False, help="Toggle between demo and real Dhan API data")
    show_emas = st.checkbox("Show EMA Lines", value=True)
    auto_refresh = st.selectbox("Auto Refresh", ["Off", "30s", "1min", "5min"])
    
    st.markdown("---")
    st.markdown("**📊 About**")
    st.markdown("This app tracks options straddle premiums in real-time, for mobile trading.")
    
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

# Enhanced Input form with IVP and recommendations
with st.form("trading_params", clear_on_submit=False):
    col1, col2, col3 = st.columns([1, 1, 1])
    
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
    
    with col3:
        # IVP Input
        ivp = st.number_input(
            "📈 IV Percentile [Please paste from Broker Terminal]",
            value=50,
            min_value=0,
            max_value=100,
            step=5,
            help="Current Implied Volatility Percentile (0-100). Affects recommendation scoring."
        )
        
    # Additional options in a second row
    col4, col5 = st.columns([1, 1])
    
    with col4:
        show_prev_close = st.checkbox(
            "📉 Previous Close Line",
            value=True
        )
        
    with col5:
        show_recommendations = st.checkbox(
            "📊 Show Recommendations",
            value=True,
            help="Display option selling recommendations table"
        )
    
    # Generate button
    submitted = st.form_submit_button(
        "🚀 Generate Chart & Recommendations",
        use_container_width=True
    )

# Main logic
if submitted:
    with st.spinner("🔄 Fetching options data..."):
        
        # Get current ticker and option data from session state
        current_ticker = st.session_state.get('current_ticker', ticker)
        current_option_data = st.session_state.get('current_option_data', option_data)
        
        # Calculate Days to Expiry (DTE)
        dte = 30  # Default fallback
        if current_option_data and 'expiry' in current_option_data:
            try:
                expiry_date = datetime.strptime(current_option_data['expiry'], '%Y-%m-%d').date()
                current_date = datetime.now().date()
                dte = max(1, (expiry_date - current_date).days + 1)  # Add 1 as instructed
                st.info(f"📅 Days to Expiry: {dte} | IV Percentile: {ivp}% | Environment: {'HIGH' if ivp >= 60 else 'MODERATE' if ivp >= 40 else 'LOW'}")
                
                # Store IVP in session state for full table access
                st.session_state['current_ivp'] = ivp
            except:
                st.warning("⚠️ Could not calculate DTE, using default value of 30 days")
        
        # Initialize client
        dhan_client = init_dhan_client()
        
        try:
            # Generate chart data
            df_data = None
            if dhan_client:
                df_data = get_real_options_data(current_ticker, ltp, agg_tick, dhan_client)
                if df_data is None:
                    st.error("❌ Could not fetch chart data. Please check your inputs and try again.")
            else:
                st.error("❌ Dhan client not initialized. Please check your API credentials.")
            
            # Create two columns for chart and quick recommendations
            if show_recommendations and current_option_data and 'option_chain' in current_option_data:
                # Show both chart and recommendations
                chart_col, rec_col = st.columns([3, 2])
                
                with chart_col:
                    st.subheader("📈 Straddle Premium Chart")
                    if df_data is not None:
                        fig = create_static_chart(df_data, current_ticker, ltp, show_prev_close, show_emas)
                        st.pyplot(fig)
                    else:
                        st.warning("Chart data not available")
                
                with rec_col:
                    # Generate recommendations using option chain data
                    with st.spinner("🔄 Calculating recommendations..."):
                        enhanced_df = process_option_chain_for_recommendations(
                            current_option_data['option_chain'], 
                            current_option_data['current_price'], 
                            ivp, 
                            dte
                        )
                        
                        if enhanced_df is not None and not enhanced_df.empty:
                            # Display compact recommendations for mobile
                            st.subheader("📊 Quick Recommendations")
                            
                            # Show top 3 calls and puts
                            calls_df = enhanced_df[enhanced_df['type'] == 'CE'].copy()
                            puts_df = enhanced_df[enhanced_df['type'] == 'PE'].copy()
                            
                            # Filter for good recommendations
                            safe_calls = calls_df[
                                (calls_df['recommendation'].isin(['STRONG_SELL', 'SELL'])) & 
                                (calls_df['delta'].abs() <= 0.5)
                            ].head(3)
                            safe_puts = puts_df[
                                (puts_df['recommendation'].isin(['STRONG_SELL', 'SELL'])) & 
                                (puts_df['delta'].abs() <= 0.5)
                            ].head(3)
                            good_calls = safe_calls
                            good_puts = safe_puts
                            
                            st.markdown("**🔥 Top Call Sells:**")
                            if not good_calls.empty:
                                for _, row in good_calls.iterrows():
                                    color = "🟢" if row['recommendation'] == 'STRONG_SELL' else "🔵"
                                    st.markdown(f"{color} {int(row['strike'])} CE: ₹{row['last_price']:.1f} | θ/Δ: {row['theta_delta_ratio']:.1f} | Score: {row['selling_score']:.0f}")
                            else:
                                st.info("No strong call selling opportunities")
                            
                            st.markdown("**🔥 Top Put Sells:**")
                            if not good_puts.empty:
                                for _, row in good_puts.iterrows():
                                    color = "🟢" if row['recommendation'] == 'STRONG_SELL' else "🔵"
                                    st.markdown(f"{color} {int(row['strike'])} PE: ₹{row['last_price']:.1f} | θ/Δ: {row['theta_delta_ratio']:.1f} | Score: {row['selling_score']:.0f}")
                            else:
                                st.info("No strong put selling opportunities")
                        else:
                            st.warning("⚠️ Could not generate recommendations")
                
                # Display full enhanced option chain table below the chart
                st.markdown("---")
                if enhanced_df is not None and not enhanced_df.empty:
                    display_recommendations_table(enhanced_df, current_option_data['current_price'])
            
            else:
                # Show only chart (original behavior)
                st.subheader("📈 Straddle Premium Chart")
                if df_data is not None:
                    fig = create_static_chart(df_data, current_ticker, ltp, show_prev_close, show_emas)
                    st.pyplot(fig)
                else:
                    st.warning("Chart data not available")
            
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")
            st.error("Please check your inputs and try again.")





# Footer
st.markdown("""
<div style='text-align: center; color: #666; font-size: 0.9em; margin: 2rem 0;'>
🛠️ Built with Streamlit | 📊 Real-time Options Tracking | 📱 An App by Pappupedia
</div>
""", unsafe_allow_html=True) 