import streamlit as st
import ccxt
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas_ta as ta
from datetime import datetime
import time

st.set_page_config(page_title="Personal Binance Futures Signals", layout="wide", page_icon="📈")

st.title("Personal Binance Futures Signal Engine")
st.markdown("**Strict confirmation-based • Real-time Binance USDⓈ-M Futures • Personal use only**")

# ────────────────────────────────────────────────
# CONFIGURATION
# ────────────────────────────────────────────────

@st.cache_resource
def get_exchange():
    return ccxt.binance({
        'enableRateLimit': True,
        'options': {'defaultType': 'future'}
    })

exchange = get_exchange()

# Sidebar settings
st.sidebar.header("Configuration")

account_balance = st.sidebar.number_input(
    "Account Balance (USDT)",
    min_value=100.0,
    value=10000.0,
    step=100.0
)

risk_pct = st.sidebar.slider(
    "Risk per Trade (%)",
    min_value=0.5,
    max_value=5.0,
    value=2.0,
    step=0.5
) / 100.0

timeframes = ['5m', '15m', '30m', '1h', '4h']
tf = st.sidebar.selectbox("Signal Timeframe", timeframes, index=2)
higher_tf = st.sidebar.selectbox("Higher-TF Trend Filter", timeframes, index=4)

rsi_oversold = st.sidebar.slider("RSI Oversold Threshold (Long)", 10, 35, 20)
rsi_overbought = st.sidebar.slider("RSI Overbought Threshold (Short)", 65, 90, 75)

default_watchlist = ['BTC/USDT:USDT', 'ETH/USDT:USDT', 'SOL/USDT:USDT', 'XRP/USDT:USDT', 'BNB/USDT:USDT']
watchlist = st.sidebar.multiselect("Watchlist", options=default_watchlist, default=default_watchlist)

show_fib = st.sidebar.checkbox("Show Fibonacci Retracement Levels", value=True)
auto_refresh = st.sidebar.checkbox("Auto-refresh every 10 seconds", value=True)

if st.sidebar.button("🔄 Manual Refresh", type="primary"):
    st.rerun()

# ────────────────────────────────────────────────
# DATA FETCHING
# ────────────────────────────────────────────────

def fetch_ohlcv(symbol: str, timeframe: str, limit: int = 400):
    try:
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        return df
    except Exception as e:
        st.error(f"Data fetch error for {symbol} ({timeframe}): {str(e)}")
        return pd.DataFrame()

def get_support_resistance(df: pd.DataFrame, window: int = 40):
    if len(df) < window:
        return None, None
    support = df['low'][-window:].min()
    resistance = df['high'][-window:].max()
    return support, resistance

# ────────────────────────────────────────────────
# SIGNAL GENERATION (strict checklist)
# ────────────────────────────────────────────────

def generate_signal(symbol: str, tf: str, higher_tf: str, balance: float, risk_ratio: float):
    df = fetch_ohlcv(symbol, tf, 300)
    df_htf = fetch_ohlcv(symbol, higher_tf, 200)

    if df.empty or len(df) < 100 or df_htf.empty:
        return {"type": "WAIT", "reason": "Insufficient historical data"}

    # Indicators
    df['rsi'] = ta.rsi(df['close'], length=14)
    df['ema50'] = ta.ema(df['close'], length=50)
    df['ema200'] = ta.ema(df['close'], length=200)

    df_htf['ema50_htf'] = ta.ema(df_htf['close'], length=50)
    df_htf['ema200_htf'] = ta.ema(df_htf['close'], length=200)

    latest = df.iloc[-1]
    prev = df.iloc[-2] if len(df) > 1 else latest
    htf_latest = df_htf.iloc[-1]

    price = latest['close']
    support, resistance = get_support_resistance(df)

    rsi = latest['rsi']

    # A) RSI zone check
    if 50 <= rsi <= 55:
        return {"type": "WAIT", "reason": "RSI in neutral zone (50–55)"}

    potential_long = rsi < rsi_oversold
    potential_short = rsi > rsi_overbought

    if not (potential_long or potential_short):
        return {"type": "WAIT", "reason": "RSI not in extreme zone"}

    # EMA trend filter (higher TF)
    htf_bull = htf_latest['ema50_htf'] > htf_latest['ema200_htf']
    htf_bear = htf_latest['ema50_htf'] < htf_latest['ema200_htf']

    # Structure proximity
    near_support = support is not None and abs(price - support) / support <= 0.008
    near_resistance = resistance is not None and abs(price - resistance) / resistance <= 0.008

    # Volume confirmation
    avg_vol = df['volume'][-30:].mean()
    vol_confirm = latest['volume'] > avg_vol * 1.5

    # Price action (simple patterns)
    hammer = (latest['close'] > latest['open']) and ((latest['high'] - latest['close']) < 0.3 * (latest['close'] - latest['open']))
    engulfing_bull = (latest['close'] > latest['open']) and (prev['close'] < prev['open']) and (latest['close'] > prev['open'])
    shooting_star = (latest['close'] < latest['open']) and ((latest['close'] - latest['low']) < 0.3 * (latest['open'] - latest['close']))
    engulfing_bear = (latest['close'] < latest['open']) and (prev['close'] > prev['open']) and (latest['close'] < prev['open'])

    pa_long = hammer or engulfing_bull
    pa_short = shooting_star or engulfing_bear

    # Checklist
    checklist = []
    passed = True

    if potential_long:
        checklist.append("RSI oversold → PASS")
        if not near_support:
            checklist.append("Near support → FAIL")
            passed = False
        if not vol_confirm:
            checklist.append("Volume spike → FAIL")
            passed = False
        if not pa_long:
            checklist.append("Price action confirmation → FAIL")
            passed = False
        if htf_bull:
            checklist.append("Higher-TF bullish → PASS")
        else:
            if rsi < 15 and vol_confirm and pa_long:
                checklist.append("Strong counter-trend reversal → PASS")
            else:
                checklist.append("Higher-TF alignment → FAIL")
                passed = False

    else:  # potential_short
        checklist.append("RSI overbought → PASS")
        if not near_resistance:
            checklist.append("Near resistance → FAIL")
            passed = False
        if not vol_confirm:
            checklist.append("Volume spike → FAIL")
            passed = False
        if not pa_short:
            checklist.append("Price action confirmation → FAIL")
            passed = False
        if htf_bear:
            checklist.append("Higher-TF bearish → PASS")
        else:
            if rsi > 85 and vol_confirm and pa_short:
                checklist.append("Strong counter-trend reversal → PASS")
            else:
                checklist.append("Higher-TF alignment → FAIL")
                passed = False

    if not passed:
        fails = " | ".join([c for c in checklist if "FAIL" in c])
        return {"type": "WAIT", "reason": fails or "Checklist incomplete", "checklist": checklist}

    # ── Valid signal ───────────────────────────────────────
    entry = price
    if potential_long:
        sl = support * 0.992 if support else entry * 0.96
        risk_dist = entry - sl
        tp1 = entry + 1.5 * risk_dist
        tp2 = entry + 2.5 * risk_dist
        tp3 = entry + 4.0 * risk_dist
    else:
        sl = resistance * 1.008 if resistance else entry * 1.04
        risk_dist = sl - entry
        tp1 = entry - 1.5 * risk_dist
        tp2 = entry - 2.5 * risk_dist
        tp3 = entry - 4.0 * risk_dist

    risk_amount = balance * risk_ratio
    risk_fraction = risk_dist / entry
    if risk_fraction <= 0:
        return {"type": "WAIT", "reason": "Invalid risk distance"}

    position_usdt = risk_amount / risk_fraction
    quantity = position_usdt / entry

    suggested_lev = max(3, min(20, int(40 / (risk_fraction * 100))))
    rr_tp1 = 1.5

    if rr_tp1 < 1.5:
        return {"type": "WAIT", "reason": "Risk:Reward < 1.5:1 on TP1"}

    confidence = 80
    if rsi < 15 or rsi > 85: confidence += 12
    if vol_confirm: confidence += 8
    confidence = min(100, confidence)

    return {
        "type": "LONG" if potential_long else "SHORT",
        "entry": round(entry, 4),
        "tp1": round(tp1, 4),
        "tp2": round(tp2, 4),
        "tp3": round(tp3, 4),
        "sl": round(sl, 4),
        "leverage": suggested_lev,
        "position_usdt": round(position_usdt, 2),
        "quantity": round(quantity, 4),
        "rr": "1:1.5 / 1:2.5 / 1:4",
        "confidence": confidence,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "checklist": checklist,
        "explanation": "All mandatory confirmations aligned."
    }

# ────────────────────────────────────────────────
# SESSION STATE
# ────────────────────────────────────────────────

if "signal_history" not in st.session_state:
    st.session_state.signal_history = []
if "active_signals" not in st.session_state:
    st.session_state.active_signals = []

# ────────────────────────────────────────────────
# LAYOUT ─ TABS
# ────────────────────────────────────────────────

tab_dashboard, tab_chart, tab_history, tab_backtest = st.tabs(
    ["📊 Dashboard", "📈 Enhanced Chart", "📜 History", "🔬 Backtest"]
)

# ── Dashboard ──────────────────────────────────────────
with tab_dashboard:
    st.subheader("Watchlist Overview")
    rows = []
    for sym in watchlist:
        sig = generate_signal(sym, tf, higher_tf, account_balance, risk_pct)
        try:
            price = exchange.fetch_ticker(sym)['last']
        except:
            price = None
        rows.append({
            "Symbol": sym,
            "Price": f"${price:,.2f}" if price else "–",
            "Signal": sig["type"],
            "Confidence": f"{sig.get('confidence', '–')}%",
            "Reason": sig.get("reason") or sig.get("explanation", "–")
        })

    if rows:
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

# ── Enhanced Chart ─────────────────────────────────────
with tab_chart:
    selected = st.selectbox("Select symbol", watchlist)

    sig = generate_signal(selected, tf, higher_tf, account_balance, risk_pct)

    df = fetch_ohlcv(selected, tf, 400)
    df_htf = fetch_ohlcv(selected, higher_tf, 200)

    if not df.empty:
        df['rsi'] = ta.rsi(df['close'], length=14)
        df['ema50'] = ta.ema(df['close'], length=50)
        df['ema200'] = ta.ema(df['close'], length=200)

        if not df_htf.empty:
            df_htf['ema50_htf'] = ta.ema(df_htf['close'], length=50)
            df_htf['ema200_htf'] = ta.ema(df_htf['close'], length=200)
            df_htf_res = df_htf.reindex(df.index, method='ffill')

        # Fibonacci swing
        w = min(120, len(df) // 2)
        swing_high = df['high'][-w:].max()
        swing_low = df['low'][-w:].min()
        fib_levels = [0.0, 0.236, 0.382, 0.5, 0.618, 0.786, 1.0]
        fib_prices = {lvl: swing_high - lvl * (swing_high - swing_low) for lvl in fib_levels}

        # Plot
        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.07,
            row_heights=[0.70, 0.30],
            subplot_titles=(f"{selected}  —  {tf.upper()}", "Volume")
        )

        # Candles
        fig.add_trace(
            go.Candlestick(
                x=df.index,
                open=df['open'], high=df['high'], low=df['low'], close=df['close'],
                name="Price",
                increasing_line_color='lime',
                decreasing_line_color='red'
            ),
            row=1, col=1
        )

        # EMAs
        fig.add_trace(go.Scatter(x=df.index, y=df['ema50'], name="EMA 50", line_color='orange'), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['ema200'], name="EMA 200", line_color='blue'), row=1, col=1)

        # Higher TF EMAs
        if 'ema50_htf' in df_htf_res.columns:
            fig.add_trace(
                go.Scatter(x=df.index, y=df_htf_res['ema50_htf'],
                           name=f"EMA 50 ({higher_tf})", line=dict(color='gold', width=2.8, dash='dash')),
                row=1, col=1
            )
            fig.add_trace(
                go.Scatter(x=df.index, y=df_htf_res['ema200_htf'],
                           name=f"EMA 200 ({higher_tf})", line=dict(color='purple', width=2.8, dash='dot')),
                row=1, col=1
            )

        # Fibonacci
        if show_fib:
            for lvl, val in fib_prices.items():
                fig.add_hline(y=val, line_dash="dash", line_color="violet", opacity=0.55,
                              annotation_text=f"{lvl:.3f}", row=1, col=1)

        # S/R
        sup, res = get_support_resistance(df)
        if sup: fig.add_hline(y=sup, line_dash="dash", line_color="green", annotation_text="Support", row=1, col=1)
        if res: fig.add_hline(y=res, line_dash="dash", line_color="red", annotation_text="Resistance", row=1, col=1)

        # Volume
        vol_colors = ['lime' if o <= c else 'red' for o, c in zip(df['open'], df['close'])]
        fig.add_trace(
            go.Bar(x=df.index, y=df['volume'], name="Volume", marker_color=vol_colors, opacity=0.65),
            row=2, col=1
        )

        fig.update_layout(
            height=820,
            template="plotly_dark",
            xaxis_rangeslider_visible=False,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            margin=dict(l=50, r=30, t=60, b=50)
        )

        st.plotly_chart(fig, use_container_width=True)

        # RSI subplot
        fig_rsi = go.Figure()
        fig_rsi.add_trace(go.Scatter(x=df.index, y=df['rsi'], line_color='purple', name='RSI'))
        fig_rsi.add_hline(y=rsi_oversold, line_dash="dot", line_color="green")
        fig_rsi.add_hline(y=rsi_overbought, line_dash="dot", line_color="red")
        fig_rsi.update_layout(height=260, template="plotly_dark", margin=dict(l=50, r=30, t=20, b=40))
        st.plotly_chart(fig_rsi, use_container_width=True)

    # Signal output
    if sig["type"] in ["LONG", "SHORT"]:
        st.success(f"**{sig['type']} SIGNAL**  —  Confidence: {sig.get('confidence', '–')}%")
        cols = st.columns([1,1,1])
        with cols[0]:
            st.metric("Entry", f"${sig['entry']}")
            st.metric("Stop Loss", f"${sig['sl']}")
        with cols[1]:
            st.metric("TP1", f"${sig['tp1']}")
            st.metric("TP2", f"${sig['tp2']}")
            st.metric("TP3", f"${sig['tp3']}")
        with cols[2]:
            st.metric("Leverage", f"{sig['leverage']}×")
            st.metric("Position", f"{sig['position_usdt']} USDT")
            st.metric("R:R", sig['rr'])

        st.markdown("**Checklist status**")
        for line in sig.get("checklist", []):
            st.write(line)

        st.info(sig.get("explanation", "–"))

    else:
        st.error("**WAIT / DO NOT ENTER**")
        st.write("**Reason**: " + sig.get("reason", "Checklist not fully satisfied"))
        if "checklist" in sig:
            st.markdown("**Checklist status**")
            for line in sig["checklist"]:
                st.write(line)

# ── History & Active ───────────────────────────────────
with tab_history:
    st.subheader("Signal History")
    if st.session_state.signal_history:
        st.dataframe(pd.DataFrame(st.session_state.signal_history), use_container_width=True)
    else:
        st.info("No signals recorded yet.")

# Auto-refresh
if auto_refresh:
    time.sleep(10)
    st.rerun() 
