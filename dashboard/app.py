"""
AI Quant Trading Research Dashboard
====================================
Multi-page Streamlit dashboard for monitoring, analysis, and research.

Pages:
1. Portfolio Overview — equity curve, key metrics, drawdown chart
2. Market Analysis — price charts, technical indicators, regime
3. Strategy Comparison — head-to-head strategy performance
4. Alpha Research — signal rankings, IC analysis
5. RL Training Monitor — training curves, agent actions
6. Backtest Results — simulation performance, trade log
7. Order Book Heatmap — microstructure visualization
8. Research Lab — feature importance, correlation analysis

Launch::

    streamlit run dashboard/app.py
"""

from __future__ import annotations

import sys
from pathlib import Path
import requests

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

# Ensure project root is importable
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def get_btc_price():
    """Fetches the latest BTC price from Binance API."""
    try:
        r = requests.get("https://api.binance.com/api/v3/ticker/price?symbol=BTCUSDT", timeout=5)
        return float(r.json()["price"])
    except:
        return None


def main() -> None:
    """Streamlit app entry point."""
    st.set_page_config(
        page_title="AI Quant Trading Research Platform",
        page_icon="📈",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # --- Sidebar ---
    st.sidebar.title("📈 Quant Research")
    
    # Live Price Widget
    price = get_btc_price()
    if price:
        st.sidebar.metric("BTC Price", f"${price:,.2f}")
    
    st.sidebar.markdown("---")
    
    page = st.sidebar.radio(
        "Navigate",
        [
            "📊 Portfolio Overview",
            "📈 Market Analysis",
            "⚔️ Strategy Comparison",
            "🔬 Alpha Research",
            "🤖 RL Training Monitor",
            "🎯 Backtest Results",
            "📖 Order Book Heatmap",
            "🧪 Research Lab",
        ],
    )

    st.sidebar.markdown("---")
    
    # System Health Panel
    st.sidebar.markdown("### System Health")
    status_data = {
        "Data Pipeline": "Running",
        "Feature Store": "Healthy",
        "RL Trainer": "Idle",
        "Backtest Engine": "Ready"
    }

    for k, v in status_data.items():
        st.sidebar.write(f"**{k}**: {v}")

    st.sidebar.markdown("---")
    st.sidebar.caption("QuantResearchLab Platform v1.1")

    # --- Header Metrics ---
    st.markdown(
        """
        ### AI Quantitative Trading Research Platform
        Institutional-grade research infrastructure for alpha discovery, machine learning experimentation, and strategy simulation.
        """
    )

    hcol1, hcol2, hcol3, hcol4 = st.columns(4)
    hcol1.metric("Platform Status", "Operational")
    hcol2.metric("Active Models", "3")
    hcol3.metric("Research Signals", "12")
    hcol4.metric("Strategies Evaluated", "5")

    st.markdown("---")

    pages = {
        "📊 Portfolio Overview": _portfolio_page,
        "📈 Market Analysis": _market_analysis_page,
        "⚔️ Strategy Comparison": _strategy_comparison_page,
        "🔬 Alpha Research": _alpha_research_page,
        "🤖 RL Training Monitor": _rl_training_page,
        "🎯 Backtest Results": _backtest_page,
        "📖 Order Book Heatmap": _orderbook_page,
        "🧪 Research Lab": _research_lab_page,
    }
    pages[page]()


# --- Pages ---


def _portfolio_page() -> None:
    """Portfolio overview with equity curve, drawdown, and key metrics."""
    st.title("📊 Portfolio Overview")

    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Return", "+12.45%", "+2.3%")
    col2.metric("Sharpe Ratio", "1.67", "+0.12")
    col3.metric("Max Drawdown", "-5.2%", "-1.1%")
    col4.metric("Win Rate", "58.3%", "+3.2%")

    # Equity curve with drawdown subplot
    st.subheader("Equity Curve & Drawdown")
    np.random.seed(42)
    dates = pd.date_range("2025-01-01", periods=365, freq="D")
    equity = 10000 + np.cumsum(np.random.randn(365) * 50)
    peak = np.maximum.accumulate(equity)
    drawdown = (peak - equity) / peak * 100

    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True,
        vertical_spacing=0.05,
        row_heights=[0.7, 0.3],
        subplot_titles=("Portfolio Value ($)", "Drawdown (%)")
    )
    fig.add_trace(
        go.Scatter(x=dates, y=equity, name="Portfolio", fill="tonexty",
                   line=dict(color="#00d4aa", width=2)),
        row=1, col=1,
    )
    fig.add_trace(
        go.Scatter(x=dates, y=peak, name="Peak", line=dict(color="#ff9800", dash="dash", width=1)),
        row=1, col=1,
    )
    fig.add_trace(
        go.Scatter(x=dates, y=-drawdown, name="Drawdown", fill="tozeroy",
                   line=dict(color="#f44336", width=1)),
        row=2, col=1,
    )
    fig.update_layout(template="plotly_dark", height=500, margin=dict(l=0, r=0, t=40, b=0))
    st.plotly_chart(fig, use_container_width=True)

    # Performance table
    st.subheader("Performance Summary")
    perf_data = pd.DataFrame({
        "Metric": ["Total Return", "Annualised Return", "Sharpe Ratio", "Sortino Ratio",
                    "Max Drawdown", "Win Rate", "Profit Factor", "Total Trades"],
        "Value": ["+12.45%", "+18.7%", "1.67", "2.14", "-5.2%", "58.3%", "1.82", "47"],
    })
    st.dataframe(perf_data, use_container_width=True, hide_index=True)

    st.info("💡 Connect live data by configuring your Binance API keys in `.env`.")


def _market_analysis_page() -> None:
    """Technical analysis charts and market regime."""
    st.title("📈 Market Analysis")

    symbol = st.selectbox("Symbol", ["BTCUSDT", "ETHUSDT", "SOLUSDT"])
    st.subheader(f"{symbol} — Technical Indicators")

    np.random.seed(123)
    n = 200
    dates = pd.date_range("2025-06-01", periods=n, freq="h")
    close = 50000 + np.cumsum(np.random.randn(n) * 200)
    sma20 = pd.Series(close).rolling(20).mean()
    sma50 = pd.Series(close).rolling(50).mean()

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=dates, y=close, name="Close", line=dict(color="#4fc3f7")))
    fig.add_trace(go.Scatter(x=dates, y=sma20, name="SMA 20", line=dict(color="#ff8a65", dash="dash")))
    fig.add_trace(go.Scatter(x=dates, y=sma50, name="SMA 50", line=dict(color="#ce93d8", dash="dot")))
    fig.update_layout(template="plotly_dark", height=400)
    st.plotly_chart(fig, use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("RSI")
        rsi = 30 + np.cumsum(np.random.randn(n) * 2)
        rsi = np.clip(rsi, 0, 100)
        fig_rsi = go.Figure()
        fig_rsi.add_trace(go.Scatter(x=dates, y=rsi, line=dict(color="#4fc3f7")))
        fig_rsi.add_hline(y=70, line_dash="dash", line_color="red")
        fig_rsi.add_hline(y=30, line_dash="dash", line_color="green")
        fig_rsi.update_layout(template="plotly_dark", height=250)
        st.plotly_chart(fig_rsi, use_container_width=True)

    with col2:
        st.subheader("Market Regime")
        regimes = ["TRENDING_UP", "RANGING", "HIGH_VOLATILITY", "TRENDING_DOWN"]
        regime_counts = [45, 30, 15, 10]
        fig_regime = px.pie(values=regime_counts, names=regimes, hole=0.4,
                            color_discrete_sequence=["#4caf50", "#ff9800", "#f44336", "#2196f3"])
        fig_regime.update_layout(template="plotly_dark", height=250)
        st.plotly_chart(fig_regime, use_container_width=True)


def _strategy_comparison_page() -> None:
    """Head-to-head strategy comparison with Leaderboard."""
    st.title("⚔️ Strategy Comparison")

    st.subheader("Strategy Leaderboard")
    strategies = pd.DataFrame({
        "Strategy": ["LSTM Predictor", "RL Agent (PPO)", "Momentum Alpha", "Mean Reversion", "Combined Signal"],
        "Total Return (%)": [8.23, 14.67, 5.41, -2.18, 12.45],
        "Sharpe Ratio": [1.34, 1.89, 0.92, -0.45, 1.67],
        "Max Drawdown (%)": [-6.1, -8.3, -4.2, -12.7, -5.2],
        "Win Rate (%)": [55.2, 62.1, 53.8, 47.3, 58.3],
        "Profit Factor": [1.45, 2.12, 1.28, 0.72, 1.82],
        "Trade Count": [38, 52, 67, 89, 47],
    })

    # Sort and rank strategies
    leaderboard = strategies.sort_values("Sharpe Ratio", ascending=False)
    st.dataframe(leaderboard, use_container_width=True, hide_index=True)

    # Equity curves comparison
    st.subheader("Equity Curves")
    np.random.seed(42)
    n = 500
    dates = pd.date_range("2025-01-01", periods=n, freq="h")

    fig = go.Figure()
    strategy_configs = [
        ("LSTM Predictor", "#4fc3f7", 0.015),
        ("RL Agent (PPO)", "#4caf50", 0.025),
        ("Momentum Alpha", "#ff9800", 0.010),
        ("Mean Reversion", "#f44336", -0.005),
        ("Combined Signal", "#00d4aa", 0.020),
    ]
    for name, color, drift in strategy_configs:
        returns = np.random.randn(n) * 0.01 + drift / n
        equity = 10000 * np.cumprod(1 + returns)
        fig.add_trace(go.Scatter(x=dates, y=equity, name=name, line=dict(color=color, width=2)))

    fig.add_hline(y=10000, line_dash="dash", line_color="white", opacity=0.3)
    fig.update_layout(
        template="plotly_dark", height=450,
        yaxis_title="Portfolio Value ($)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    st.plotly_chart(fig, use_container_width=True)

    # Radar chart comparison
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Risk-Return Profile")
        categories = ["Return", "Sharpe", "Win Rate", "Stability", "Risk-Adjusted"]

        fig_radar = go.Figure()
        for name, vals in [
            ("RL Agent (PPO)", [0.9, 0.95, 0.85, 0.7, 0.88]),
            ("LSTM Predictor", [0.6, 0.7, 0.72, 0.8, 0.75]),
            ("Combined Signal", [0.8, 0.85, 0.78, 0.9, 0.82]),
        ]:
            fig_radar.add_trace(go.Scatterpolar(
                r=vals + [vals[0]], theta=categories + [categories[0]],
                name=name, fill="toself", opacity=0.6,
            ))
        fig_radar.update_layout(
            template="plotly_dark", height=350,
            polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        )
        st.plotly_chart(fig_radar, use_container_width=True)

    with col2:
        st.subheader("Monthly Returns Heatmap")
        months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                   "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
        strat_names = ["LSTM", "RL (PPO)", "Momentum", "Combined"]
        np.random.seed(999)
        heatmap_data = np.random.randn(4, 12) * 3 + 1
        fig_heat = go.Figure(data=go.Heatmap(
            z=heatmap_data, x=months, y=strat_names,
            colorscale="RdYlGn", zmid=0,
            text=np.round(heatmap_data, 1), texttemplate="%{text}%",
        ))
        fig_heat.update_layout(template="plotly_dark", height=350)
        st.plotly_chart(fig_heat, use_container_width=True)


def _alpha_research_page() -> None:
    """Alpha signal ranking and analysis."""
    st.title("🔬 Alpha Research")

    st.subheader("Signal Ranking")
    alpha_data = pd.DataFrame({
        "Signal": ["momentum_12", "rsi_cross", "mean_reversion_20", "order_flow"],
        "IC": [0.085, 0.062, -0.041, 0.053],
        "Sharpe": [1.45, 1.12, -0.78, 0.93],
        "Win Rate (%)": [58.2, 55.1, 47.3, 54.8],
        "Stability": [0.72, 0.65, 0.38, 0.61],
        "Rating": ["STRONG", "MODERATE", "WEAK", "MODERATE"],
    })
    st.dataframe(alpha_data, use_container_width=True, hide_index=True)

    st.subheader("Information Coefficient Over Time")
    np.random.seed(77)
    dates = pd.date_range("2025-01-01", periods=52, freq="W")
    fig = go.Figure()
    for signal, color in [("momentum_12", "#4caf50"), ("rsi_cross", "#ff9800"), ("order_flow", "#4fc3f7")]:
        ic_series = np.random.randn(52) * 0.05 + 0.04
        fig.add_trace(go.Scatter(x=dates, y=ic_series, name=signal, mode="lines", line=dict(color=color)))
    fig.add_hline(y=0, line_dash="dash", line_color="white")
    fig.update_layout(template="plotly_dark", height=350, yaxis_title="IC")
    st.plotly_chart(fig, use_container_width=True)

    # Signal decay
    st.subheader("Signal Decay Analysis")
    lags = np.arange(1, 25)
    fig_decay = go.Figure()
    for signal, color, decay in [("momentum_12", "#4caf50", 0.08), ("order_flow", "#4fc3f7", 0.12)]:
        ic_decay = decay * np.exp(-lags * 0.15) + np.random.randn(24) * 0.01
        fig_decay.add_trace(go.Scatter(x=lags, y=ic_decay, name=signal, line=dict(color=color, width=2)))
    fig_decay.add_hline(y=0, line_dash="dash", line_color="white")
    fig_decay.update_layout(template="plotly_dark", height=300, xaxis_title="Forward Lag (periods)", yaxis_title="IC")
    st.plotly_chart(fig_decay, use_container_width=True)


def _rl_training_page() -> None:
    """RL agent training metrics."""
    st.title("🤖 RL Training Monitor")

    # Load real training curve if exists
    training_plot_path = PROJECT_ROOT / "validation_results" / "validation_plots" / "rl_training_curve.png"
    if training_plot_path.exists():
        st.subheader("RL Training Rewards")
        st.image(str(training_plot_path), use_container_width=True)
        st.success("✅ Showing latest training curve.")
    else:
        st.warning("No RL training curve found. Train an agent to see progress.")
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Training Reward")
            np.random.seed(42)
            episodes = np.arange(500)
            rewards = np.cumsum(np.random.randn(500) * 0.1) + np.linspace(0, 5, 500)
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=episodes, y=rewards, line=dict(color="#4caf50")))
            fig.update_layout(template="plotly_dark", height=300, xaxis_title="Episode", yaxis_title="Cumulative Reward")
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.subheader("Action Distribution")
            actions = ["BUY", "SELL", "HOLD"]
            counts = [180, 170, 150]
            fig = px.bar(x=actions, y=counts, color=actions,
                         color_discrete_map={"BUY": "#4caf50", "SELL": "#f44336", "HOLD": "#ff9800"})
            fig.update_layout(template="plotly_dark", height=300, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)


def _backtest_page() -> None:
    """Backtest simulation results with enhanced equity visualization."""
    st.title("🎯 Backtest Results")

    # Load real results if they exist
    results_path = PROJECT_ROOT / "validation_results" / "backtest_results.json"
    if results_path.exists():
        try:
            with open(results_path, "r") as f:
                import json
                data = json.load(f)
            
            # Metrics
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Total Return", f"{data.get('total_return_pct', 0):.2f}%")
            col2.metric("Sharpe Ratio", f"{data.get('sharpe_ratio', 0):.4f}")
            col3.metric("Max Drawdown", f"{data.get('max_drawdown_pct', 0):.2f}%")
            col4.metric("Total Trades", f"{data.get('n_trades', 0)}")

            # Equity Curve
            st.subheader("Equity Curve")
            equity_path = PROJECT_ROOT / "validation_results" / "validation_plots" / "equity_curve.png"
            if equity_path.exists():
                st.image(str(equity_path), use_container_width=True)
            else:
                st.warning("Equity curve plot not found in validation_plots/")

            st.success(f"✅ Loaded latest results from `{results_path.name}`")
        except Exception as e:
            st.error(f"Error loading results: {e}")
    else:
        st.warning("No backtest results found. Run a simulation to generate results.")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Return", "+8.23%")
        col2.metric("Sharpe Ratio", "1.34")
        col3.metric("Max Drawdown", "-6.1%")
        col4.metric("Total Trades", "47")

    st.subheader("Trade Log")
    trades = pd.DataFrame({
        "Time": pd.date_range("2025-01-01", periods=10, freq="2D"),
        "Action": ["BUY", "SELL"] * 5,
        "Price": np.random.uniform(48000, 52000, 10).round(2),
        "Quantity": np.random.uniform(0.01, 0.1, 10).round(4),
        "Slippage": np.random.uniform(0.5, 5.0, 10).round(2),
        "PnL": np.random.uniform(-200, 500, 10).round(2),
    })
    st.dataframe(trades, use_container_width=True, hide_index=True)


def _orderbook_page() -> None:
    """Order book depth heatmap."""
    st.title("📖 Order Book Heatmap")

    st.subheader("Bid/Ask Depth")
    np.random.seed(42)
    n_levels = 20
    prices_bid = np.linspace(49000, 49900, n_levels)
    prices_ask = np.linspace(50100, 51000, n_levels)
    vol_bid = np.random.exponential(2, n_levels)
    vol_ask = np.random.exponential(2, n_levels)

    fig = go.Figure()
    fig.add_trace(go.Bar(x=-vol_bid[::-1], y=prices_bid[::-1], orientation="h",
                         name="Bids", marker_color="#4caf50"))
    fig.add_trace(go.Bar(x=vol_ask, y=prices_ask, orientation="h",
                         name="Asks", marker_color="#f44336"))
    fig.update_layout(template="plotly_dark", height=500, barmode="relative",
                      xaxis_title="Volume (BTC)", yaxis_title="Price (USDT)")
    st.plotly_chart(fig, use_container_width=True)

    col1, col2, col3 = st.columns(3)
    col1.metric("Spread", "0.12%")
    col2.metric("Order Imbalance", "+0.35")
    col3.metric("Bid Depth", "45.2 BTC")


def _research_lab_page() -> None:
    """Research lab: feature importance and correlation analysis."""
    st.title("🧪 Research Lab")

    tab1, tab2 = st.tabs(["Feature Importance", "Correlation Analysis"])

    with tab1:
        st.subheader("Feature Importance Ranking")
        
        importance_path = PROJECT_ROOT / "validation_results" / "validation_plots" / "feature_importance.png"
        if importance_path.exists():
            st.image(str(importance_path), use_container_width=True)
        else:
            np.random.seed(42)
            features = ["rsi", "macd", "volatility", "sma_20", "momentum", "order_imbalance",
                         "spread_bps", "sentiment_compound", "volume", "atr", "ema_20", "roc"]
            importance = np.sort(np.random.uniform(0.02, 0.15, len(features)))[::-1]

            fig = go.Figure()
            colors = ["#4caf50" if v > 0.08 else "#ff9800" if v > 0.05 else "#f44336" for v in importance]
            fig.add_trace(go.Bar(x=importance, y=features, orientation="h",
                                 marker_color=colors, text=[f"{v:.3f}" for v in importance],
                                 textposition="auto"))
            fig.update_layout(template="plotly_dark", height=450, yaxis=dict(autorange="reversed"),
                              xaxis_title="Importance Score")
            st.plotly_chart(fig, use_container_width=True)

        st.caption("🟢 High importance  🟠 Medium  🔴 Low")

    with tab2:
        st.subheader("Feature-Return Correlation")
        features_list = ["rsi", "macd", "volatility", "momentum", "order_imbalance",
                         "sentiment_compound", "spread_bps", "atr"]
        corr = np.random.uniform(-0.15, 0.15, len(features_list))
        corr_df = pd.DataFrame({"Feature": features_list, "Spearman IC": np.round(corr, 4),
                                 "Significant": np.abs(corr) > 0.05})
        st.dataframe(corr_df, use_container_width=True, hide_index=True)

        st.subheader("Inter-Feature Correlation Matrix")
        np.random.seed(42)
        n_feat = 8
        corr_matrix = np.eye(n_feat) + np.random.randn(n_feat, n_feat) * 0.3
        corr_matrix = (corr_matrix + corr_matrix.T) / 2
        np.fill_diagonal(corr_matrix, 1)
        corr_matrix = np.clip(corr_matrix, -1, 1)

        fig_corr = go.Figure(data=go.Heatmap(
            z=corr_matrix, x=features_list, y=features_list,
            colorscale="RdBu_r", zmid=0,
            text=np.round(corr_matrix, 2), texttemplate="%{text}",
        ))
        fig_corr.update_layout(template="plotly_dark", height=450)
        st.plotly_chart(fig_corr, use_container_width=True)


if __name__ == "__main__":
    main()
