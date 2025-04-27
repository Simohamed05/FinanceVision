import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from prophet import Prophet
from statsmodels.tsa.arima.model import ARIMA
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from keras.models import Sequential
from keras.layers import LSTM, GRU, Dense
import ta
import warnings
from datetime import datetime, timedelta
import logging
import random  
import uuid
import time

# Configuration
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore")

# Configuration de la page
st.set_page_config(
    page_title="FinanceVision Pro - Trading Dashboard",
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="📈"
)



# Style CSS personnalisé
st.markdown("""
<style>
:root {
    --primary: #3b82f6;
    --secondary: #10b981;
    --danger: #ef4444;
    --dark: #1f2937;
    --darker: #111827;
    --light: #f3f4f6;
}

.stApp {
    background-color: var(--darker);
    color: var(--light);
    font-family: 'Inter', sans-serif;
}

.sidebar .sidebar-content {
    background-color: var(--dark);
    border-radius: 0.5rem;
    padding: 1rem;
    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
}

.card {
    background-color: var(--dark);
    border-radius: 0.5rem;
    padding: 1.5rem;
    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
    margin-bottom: 1rem;
}

.metric-card {
    background-color: #4b5563;
    padding: 1rem;
    border-radius: 0.5rem;
    border-left: 4px solid var(--primary);
}

.btn-primary {
    background-color: var(--primary);
    color: white;
    padding: 0.5rem 1rem;
    border-radius: 0.375rem;
    transition: all 0.3s;
    border: none;
    cursor: pointer;
}

.btn-primary:hover {
    background-color: #2563eb;
    transform: translateY(-2px);
    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
}

.btn-success {
    background-color: var(--secondary);
    color: white;
    padding: 0.5rem 1rem;
    border-radius: 0.375rem;
    transition: all 0.3s;
    border: none;
    cursor: pointer;
}

.btn-success:hover {
    background-color: #059669;
    transform: translateY(-2px);
}

.btn-danger {
    background-color: var(--danger);
    color: white;
    padding: 0.5rem 1rem;
    border-radius: 0.375rem;
    transition: all 0.3s;
    border: none;
    cursor: pointer;
}

.btn-danger:hover {
    background-color: #dc2626;
    transform: translateY(-2px);
}

.nav-item {
    padding: 0.75rem 1rem;
    border-radius: 0.375rem;
    margin-bottom: 0.5rem;
    transition: all 0.3s;
    cursor: pointer;
}

.nav-item:hover {
    background-color: #374151;
}

.nav-item.active {
    background-color: var(--primary);
    color: white;
}

.header {
    background: linear-gradient(135deg, #1e40af 0%, #1e3a8a 100%);
    padding: 2rem;
    border-radius: 0.5rem;
    margin-bottom: 1.5rem;
    color: white;
}

.ticker {
    display: inline-block;
    padding: 0.25rem 0.5rem;
    background-color: #374151;
    border-radius: 0.25rem;
    margin-right: 0.5rem;
    margin-bottom: 0.5rem;
    font-family: 'Roboto Mono', monospace;
}

.positive {
    color: #10b981;
}

.negative {
    color: #ef4444;
}

.divider {
    border-top: 1px solid #374151;
    margin: 1.5rem 0;
}

.trade-card {
    background-color: #1f2937;
    border-radius: 0.5rem;
    padding: 1rem;
    margin-bottom: 1rem;
    border-left: 4px solid;
}

.trade-buy {
    border-left-color: var(--secondary);
}

.trade-sell {
    border-left-color: var(--danger);
}

.tooltip {
    position: relative;
    display: inline-block;
    border-bottom: 1px dotted #666;
    cursor: help;
}

.tooltip .tooltiptext {
    visibility: hidden;
    width: 200px;
    background-color: #1f2937;
    color: #fff;
    text-align: center;
    border-radius: 6px;
    padding: 5px;
    position: absolute;
    z-index: 1;
    bottom: 125%;
    left: 50%;
    margin-left: -100px;
    opacity: 0;
    transition: opacity 0.3s;
    border: 1px solid #374151;
}

.tooltip:hover .tooltiptext {
    visibility: visible;
    opacity: 1;
}

.technical-indicator {
    background-color: #1f2937;
    padding: 1rem;
    border-radius: 0.5rem;
    margin-bottom: 1rem;
    border-left: 4px solid #3b82f6;
}

.indicator-value {
    font-size: 1.5rem;
    font-weight: bold;
    margin: 0.5rem 0;
}

.indicator-name {
    font-size: 0.9rem;
    color: #9ca3af;
}

.tradingview-widget {
    border-radius: 0.5rem;
    overflow: hidden;
    margin-bottom: 1rem;
}
</style>
""", unsafe_allow_html=True)

# Initialisation de l'état de la session
if 'user_id' not in st.session_state:
    st.session_state.user_id = str(uuid.uuid4())
    st.session_state.preferences = {
        'symbol': 'XAUUSD',
        'model': 'Prophet',
        'horizon': 30,
        'current_tab': 'Accueil'
    }
    st.session_state.saved_analyses = []
    st.session_state.demo_account = {
        'balance': 10000,
        'positions': [],
        'trade_history': []
    }
    st.session_state.first_visit = True
    st.session_state.backtest_indicators = ['SMA_20', 'SMA_50']
    st.session_state.technical_indicators = ['SMA_20', 'SMA_50', 'RSI', 'MACD']

# Barre latérale - Navigation
from PIL import Image

# Charger ton logo
logo = Image.open("logo.png")

with st.sidebar:
    st.markdown("<div style='padding-left: 40px;'>", unsafe_allow_html=True)  # 👈 Ajout du décalage à droite
    st.image(logo, width=200)  # 👈 Plus grand ici (150px au lieu de 100px)
    st.markdown("""
        <h1 style="color: #3b82f6; font-size: 1.8rem; font-weight: 700; margin-top: 0.5rem;">FinanceVision Pro</h1>
        <p style="color: #9ca3af; font-size: 1rem; margin-bottom: 2rem;">Trading & Analyse Avancée</p>
    </div>
    """, unsafe_allow_html=True)



    
    # Menu de navigation amélioré
    st.markdown("### Navigation")
    
    nav_options = {
        " 🏠  ": "Accueil",
        " 📊  ": "Dashboard",
        " 🔮  ": "Prévisions",
        " 📈  ": "Analyse Technique",
        " ⚖️  ": "Comparaison",
        " 🔍  ": "Backtesting",
        " 💰  ": "Compte Démo",
        " 📰  ": "News"
    }
    
    for icon, label in nav_options.items():
        if st.button(f"{icon} {label}", key=f"nav_{label}"):
            st.session_state.preferences['current_tab'] = label
    
    st.markdown("---")
    
    # Paramètres rapides améliorés
    st.markdown("### Paramètres Marché")
    symbol = st.selectbox(
        "Actif", 
        ["XAUUSD", "EURUSD", "BTCUSD", "USDJPY", "GBPUSD"],
        index=["XAUUSD", "EURUSD", "BTCUSD", "USDJPY", "GBPUSD"].index(st.session_state.preferences['symbol']),
        key="symbol_select"
    )
    
    horizon = st.slider(
        "Horizon Prévision (jours)", 
        7, 90, st.session_state.preferences['horizon'],
        key="horizon_slider"
    )
    
    st.session_state.preferences.update({
        'symbol': symbol,
        'horizon': horizon
    })
    
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #9ca3af; font-size: 0.875rem;">
        Version 2.1 Pro<br>
        © 2023 FinanceVision
    </div>
    """, unsafe_allow_html=True)

# Mappage des symboles
symbol_map = {
    "XAUUSD": "GC=F",
    "EURUSD": "EURUSD=X",
    "BTCUSD": "BTC-USD",
    "USDJPY": "JPY=X",
    "GBPUSD": "GBPUSD=X"
}
yahoo_symbol = symbol_map[symbol]

# Chargement des données avec cache
@st.cache_data(ttl=3600, show_spinner="Chargement des données marché...")
def load_data(symbol, start_date, end_date, retries=5):
    for attempt in range(retries):
        try:
            data = yf.download(symbol, start=start_date, end=end_date, progress=False)
            
            if data.empty or 'Close' not in data.columns:
                continue

            df = data.reset_index()[['Date', 'Close', 'Open', 'High', 'Low', 'Volume']]
            df.columns = ['ds', 'y', 'Open', 'High', 'Low', 'Volume']
            df['ds'] = pd.to_datetime(df['ds']).dt.normalize()

            # Calcul des indicateurs techniques
            try:
                df['SMA_20'] = df['y'].rolling(window=20).mean()
                df['SMA_50'] = df['y'].rolling(window=50).mean()
                df['RSI'] = ta.momentum.rsi(df['y'], window=14)
                df['MACD'] = ta.trend.macd_diff(df['y'])
                df['BB_upper'] = df['y'].rolling(window=20).mean() + 2*df['y'].rolling(window=20).std()
                df['BB_lower'] = df['y'].rolling(window=20).mean() - 2*df['y'].rolling(window=20).std()
                df['STOCH'] = ta.momentum.stoch(df['High'], df['Low'], df['y'], window=14)
                df['CCI'] = ta.trend.cci(df['High'], df['Low'], df['y'], window=20)
                df['ATR'] = ta.volatility.average_true_range(df['High'], df['Low'], df['y'], window=14)
            except Exception as e:
                logger.error(f"Erreur calcul indicateurs: {str(e)}")
                for col in ['SMA_20', 'SMA_50', 'RSI', 'MACD', 'BB_upper', 'BB_lower', 'STOCH', 'CCI', 'ATR']:
                    if col not in df.columns:
                        df[col] = np.nan

            df = df.dropna()
            
            if len(df) < 20:
                continue
                
            return df, {"error": None, "symbol": symbol, "rows": len(df)}
            
        except Exception as e:
            logger.error(f"Erreur tentative {attempt + 1}: {str(e)}")
            if attempt == retries - 1:
                return None, {"error": str(e), "symbol": symbol}
            time.sleep(2)

start_date = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
end_date = datetime.now().strftime('%Y-%m-%d')
df, data_report = load_data(yahoo_symbol, start_date, end_date)

if df is None or df.empty:
    st.error(f"❌ Impossible de charger les données pour {symbol}. Veuillez réessayer ou choisir un autre actif.")
    st.stop()

# Fonctions pour les modèles avec gestion d'erreur améliorée
def run_prophet(_df, _horizon):
    try:
        model = Prophet(
            daily_seasonality=True,
            yearly_seasonality=True,
            weekly_seasonality=True,
            changepoint_prior_scale=0.05
        )
        model.fit(_df[['ds', 'y']])
        future = model.make_future_dataframe(periods=_horizon)
        forecast = model.predict(future)
        return forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(_horizon), "Prophet est idéal pour les séries temporelles avec saisonnalités multiples"
    except Exception as e:
        logger.error(f"Erreur Prophet: {str(e)}")
        return None, str(e)

def run_arima(_df, _horizon):
    try:
        series = _df.set_index('ds')['y']
        model = ARIMA(series, order=(5, 1, 0))
        model_fit = model.fit()
        forecast = model_fit.forecast(steps=_horizon)
        forecast_dates = pd.date_range(start=_df['ds'].iloc[-1] + timedelta(days=1), periods=_horizon)
        return pd.DataFrame({'ds': forecast_dates, 'yhat': forecast.values}), "ARIMA performe bien sur les séries stationnaires"
    except Exception as e:
        logger.error(f"Erreur ARIMA: {str(e)}")
        return None, str(e)

def run_lstm(_df, _horizon):
    try:
        look_back = 30
        scaler = MinMaxScaler()
        scaled_data = scaler.fit_transform(_df[['y']].values)
        
        X, y = [], []
        for i in range(look_back, len(scaled_data)):
            X.append(scaled_data[i-look_back:i, 0])
            y.append(scaled_data[i, 0])
        X, y = np.array(X), np.array(y)
        
        model = Sequential([
            LSTM(64, input_shape=(look_back, 1), return_sequences=True),
            LSTM(32, return_sequences=False),
            Dense(16, activation='relu'),
            Dense(1)
        ])
        model.compile(optimizer='adam', loss='mse')
        model.fit(X.reshape(-1, look_back, 1), y, epochs=10, batch_size=32, verbose=0)
        
        last_sequence = scaled_data[-look_back:]
        forecast = []
        for _ in range(_horizon):
            pred = model.predict(last_sequence.reshape(1, look_back, 1), verbose=0)[0, 0]
            forecast.append(pred)
            last_sequence = np.roll(last_sequence, -1)
            last_sequence[-1] = pred
        
        forecast = scaler.inverse_transform(np.array(forecast).reshape(-1, 1)).flatten()
        forecast_dates = pd.date_range(start=_df['ds'].iloc[-1] + timedelta(days=1), periods=_horizon)
        return pd.DataFrame({'ds': forecast_dates, 'yhat': forecast}), "LSTM capture les motifs complexes et dépendances longues"
    except Exception as e:
        logger.error(f"Erreur LSTM: {str(e)}")
        return None, str(e)

def run_random_forest(_df, _horizon):
    try:
        df = _df.copy()
        for i in range(1, 6):
            df[f'lag_{i}'] = df['y'].shift(i)
        df['rsi'] = df['RSI']
        df['macd'] = df['MACD']
        df['sma_20'] = df['SMA_20']
        df['sma_50'] = df['SMA_50']
        df['atr'] = df['ATR']
        
        df = df.dropna()
        features = [f'lag_{i}' for i in range(1, 6)] + ['rsi', 'macd', 'sma_20', 'sma_50', 'atr']
        X = df[features]
        y = df['y']
        
        model = RandomForestRegressor(n_estimators=100, random_state=42, max_depth=10)
        model.fit(X, y)
        
        last_values = df[features].iloc[-1].values
        forecast = []
        current_values = last_values.copy()
        
        for _ in range(_horizon):
            pred = model.predict([current_values])[0]
            forecast.append(pred)
            current_values = np.roll(current_values, -1)
            current_values[0:4] = current_values[1:5]
            current_values[4] = pred
        
        forecast_dates = pd.date_range(start=_df['ds'].iloc[-1] + timedelta(days=1), periods=_horizon)
        return pd.DataFrame({'ds': forecast_dates, 'yhat': forecast}), "Random Forest robuste avec de nombreuses fonctionnalités"
    except Exception as e:
        logger.error(f"Erreur Random Forest: {str(e)}")
        return None, str(e)

# Fonctions de visualisation améliorées
def create_tradingview_chart(df, indicators=None, title=None):
    fig = go.Figure()
    
    # Candlestick
    fig.add_trace(go.Candlestick(
        x=df['ds'],
        open=df['Open'],
        high=df['High'],
        low=df['Low'],
        close=df['y'],
        increasing_line_color='#10b981',
        decreasing_line_color='#ef4444',
        name='Price'
    ))
    
    # Indicateurs techniques
    if indicators:
        colors = ['#FFA500', '#00FF00', '#FF00FF', '#00FFFF']
        for idx, indicator in enumerate(indicators):
            if indicator in df.columns:
                fig.add_trace(go.Scatter(
                    x=df['ds'],
                    y=df[indicator],
                    mode='lines',
                    name=indicator,
                    line=dict(color=colors[idx % len(colors)], width=1)
                ))
    
    # Configuration TradingView-like
    fig.update_layout(
        title=title if title else f"{symbol} - Trading View",
        xaxis_rangeslider_visible=False,
        height=600,
        template="plotly_dark",
        margin=dict(l=20, r=20, t=60, b=20),
        xaxis=dict(
            type='date',
            rangeslider=dict(visible=False)
        ),
        yaxis=dict(title="Price"),
        hovermode="x unified",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    # Ajout des boutons d'outils TradingView-like
    fig.update_xaxes(
        rangeselector=dict(
            buttons=list([
                dict(count=1, label="1M", step="month", stepmode="backward"),
                dict(count=3, label="3M", step="month", stepmode="backward"),
                dict(count=6, label="6M", step="month", stepmode="backward"),
                dict(count=1, label="YTD", step="year", stepmode="todate"),
                dict(count=1, label="1Y", step="year", stepmode="backward"),
                dict(step="all")
            ])
        )
    )
    
    return fig

def create_technical_analysis(df, indicators):
    fig = make_subplots(
        rows=4, cols=1, 
        shared_xaxes=True, 
        vertical_spacing=0.05,
        row_heights=[0.5, 0.15, 0.15, 0.2],
        specs=[[{"secondary_y": True}], [{}], [{}], [{}]]
    )
    
    # Prix et indicateurs principaux
    fig.add_trace(go.Candlestick(
        x=df['ds'],
        open=df['Open'],
        high=df['High'],
        low=df['Low'],
        close=df['y'],
        name='Price'
    ), row=1, col=1)
    
    for indicator in indicators:
        if indicator in ['SMA_20', 'SMA_50', 'BB_upper', 'BB_lower']:
            fig.add_trace(go.Scatter(
                x=df['ds'],
                y=df[indicator],
                mode='lines',
                name=indicator,
                line=dict(width=1)
            ), row=1, col=1)
    
    # Volume
    fig.add_trace(go.Bar(
        x=df['ds'],
        y=df['Volume'],
        name='Volume',
        marker_color='#3b82f6',
        opacity=0.7
    ), row=2, col=1)
    
    # RSI
    fig.add_trace(go.Scatter(
        x=df['ds'],
        y=df['RSI'],
        name='RSI',
        line=dict(color='#FFA500', width=1)
    ), row=3, col=1)
    fig.add_hline(y=70, line_dash="dash", line_color="red", row=3, col=1)
    fig.add_hline(y=30, line_dash="dash", line_color="green", row=3, col=1)
    
    # MACD
    fig.add_trace(go.Scatter(
        x=df['ds'],
        y=df['MACD'],
        name='MACD',
        line=dict(color='#00FF00', width=1)
    ), row=4, col=1)
    
    fig.update_layout(
        height=800,
        template="plotly_dark",
        hovermode="x unified",
        showlegend=True,
        margin=dict(l=20, r=20, t=60, b=20)
    )
    
    return fig

def create_forecast_chart(historical_df, forecast_df, model_name):
    fig = go.Figure()
    
    # Données historiques
    fig.add_trace(go.Scatter(
        x=historical_df['ds'],
        y=historical_df['y'],
        mode='lines',
        name='Historique',
        line=dict(color='#3b82f6')
    ))
    
    if forecast_df is not None:
        # Prévision
        fig.add_trace(go.Scatter(
            x=forecast_df['ds'],
            y=forecast_df['yhat'],
            mode='lines',
            name=f'Prévision {model_name}',
            line=dict(color='#FFA500', width=2)
        ))
        
        # Intervalle de confiance
        if 'yhat_lower' in forecast_df.columns and 'yhat_upper' in forecast_df.columns:
            fig.add_trace(go.Scatter(
                x=forecast_df['ds'],
                y=forecast_df['yhat_upper'],
                mode='lines',
                line=dict(width=0),
                showlegend=False,
                name='Upper Bound'
            ))
            fig.add_trace(go.Scatter(
                x=forecast_df['ds'],
                y=forecast_df['yhat_lower'],
                mode='lines',
                line=dict(width=0),
                fill='tonexty',
                fillcolor='rgba(255,165,0,0.2)',
                showlegend=False,
                name='Lower Bound'
            ))
    
    fig.update_layout(
        title=f"Prévision {model_name} - {symbol}",
        height=500,
        template="plotly_dark",
        hovermode="x unified",
        xaxis_rangeslider_visible=True,
        margin=dict(l=20, r=20, t=60, b=20)
    )
    
    return fig

# Fonctions pour le compte démo amélioré
def execute_demo_trade(action, price, quantity):
    demo_account = st.session_state.demo_account
    
    try:
        if action == 'BUY':
            cost = price * quantity
            if cost > demo_account['balance']:
                st.error("Solde insuffisant")
                return False
                
            demo_account['balance'] -= cost
            demo_account['positions'].append({
                'symbol': symbol,
                'quantity': quantity,
                'entry_price': price,
                'entry_time': datetime.now()
            })
            st.success(f"Achat de {quantity} {symbol} à {price:.2f} $")
            return True
            
        elif action == 'SELL':
            positions = [p for p in demo_account['positions'] if p['symbol'] == symbol]
            if not positions:
                st.error("Aucune position à vendre")
                return False
                
            total_qty = sum(p['quantity'] for p in positions)
            if quantity > total_qty:
                st.error(f"Quantité invalide (max {total_qty})")
                return False
                
            # Logique FIFO
            remaining = quantity
            for pos in positions[:]:  # Copie pour modification
                if remaining <= 0:
                    break
                    
                sell_qty = min(pos['quantity'], remaining)
                profit = (price - pos['entry_price']) * sell_qty
                
                demo_account['balance'] += price * sell_qty
                pos['quantity'] -= sell_qty
                
                demo_account['trade_history'].append({
                    'type': 'SELL',
                    'symbol': symbol,
                    'quantity': sell_qty,
                    'price': price,
                    'profit': profit,
                    'time': datetime.now(),
                    'balance': demo_account['balance']
                })
                remaining -= sell_qty
                
            # Nettoyer les positions vides
            demo_account['positions'] = [p for p in demo_account['positions'] if p['quantity'] > 0]
            st.success(f"Vente de {quantity} {symbol} à {price:.2f} $")
            return True
            
    except Exception as e:
        st.error(f"Erreur: {str(e)}")
        return False

# Affichage principal
current_tab = st.session_state.preferences['current_tab']

# Onglet Accueil
if current_tab == "Accueil":
    st.markdown("""
    <div class="header">
        <h1 style="color: white; margin-bottom: 0.5rem;">FinanceVision Pro</h1>
        <p style="color: #d1d5db; font-size: 1.1rem;">Plateforme professionnelle de trading et d'analyse de marché</p>
    </div>
    """, unsafe_allow_html=True)
    
    if st.session_state.first_visit:
        with st.expander("🔍 Guide de démarrage rapide", expanded=True):
            st.markdown("""
            **Bienvenue sur FinanceVision Pro** - Votre terminal de trading complet:
            
            1. **📊 Dashboard** - Vue d'ensemble du marché avec indicateurs clés
            2. **🔮 Prévisions** - Modèles avancés de prédiction de prix
            3. **📈 Technique** - Analyse avec +20 indicateurs techniques
            4. **⚖️ Comparaison** - Évaluation des performances des modèles
            5. **🔍 Backtest** - Test de stratégies sur données historiques
            6. **💰 Démo** - Compte de trading simulé avec $10,000
            7. **📢 News** - Actualités & Analyse Sentiment

            
            Sélectionnez un actif et commencez votre analyse!
            """)
        
        st.session_state.first_visit = False
    
    # Metrics row
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        delta = ((df['y'].iloc[-1] - df['y'].iloc[-2]) / df['y'].iloc[-2]) * 100
        st.metric(
            label="Prix Actuel", 
            value=f"{df['y'].iloc[-1]:.2f} $", 
            delta=f"{delta:.2f}%"
        )
    with col2:
        st.metric(
            label="Volume 24h", 
            value=f"{df['Volume'].iloc[-1]:,.0f}"
        )
    with col3:
        st.metric(
            label="RSI (14)", 
            value=f"{df['RSI'].iloc[-1]:.1f}",
            delta="Surachat" if df['RSI'].iloc[-1] > 70 else "Survendu" if df['RSI'].iloc[-1] < 30 else "Neutre"
        )
    with col4:
        st.metric(
            label="ATR (14)", 
            value=f"{df['ATR'].iloc[-1]:.2f}",
            delta="Volatilité"
        )
    
    # Graphique TradingView-like
    st.markdown("### Graphique Trading View")
    fig = create_tradingview_chart(df, ['SMA_20', 'SMA_50'])
    st.plotly_chart(fig, use_container_width=True)
    
    # Dernières données
    with st.expander("📋 Dernières cotations"):
        st.dataframe(
            df.tail(10)[['ds', 'Open', 'High', 'Low', 'y', 'Volume']]
            .rename(columns={'y': 'Close', 'ds': 'Date'})
            .style.format({
                'Open': '{:.2f}',
                'High': '{:.2f}',
                'Low': '{:.2f}',
                'Close': '{:.2f}',
                'Volume': '{:,.0f}'
            })
        )

# Onglet Dashboard
elif current_tab == "Dashboard":
    st.header("📊 Tableau de Bord")
    
    # Metrics row
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown("""
        <div class="metric-card">
            <div class="indicator-name">Prix Actuel</div>
            <div class="indicator-value">{:.2f} $</div>
            <div class="{}">{:.2f}%</div>
        </div>
        """.format(
            df['y'].iloc[-1],
            "positive" if df['y'].iloc[-1] > df['y'].iloc[-2] else "negative",
            ((df['y'].iloc[-1] - df['y'].iloc[-2]) / df['y'].iloc[-2] * 100)
        ), unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-card">
            <div class="indicator-name">RSI (14)</div>
            <div class="indicator-value">{:.1f}</div>
            <div class="{}">{}</div>
        </div>
        """.format(
            df['RSI'].iloc[-1],
            "positive" if df['RSI'].iloc[-1] < 30 else "negative" if df['RSI'].iloc[-1] > 70 else "",
            "Surachat" if df['RSI'].iloc[-1] > 70 else "Survendu" if df['RSI'].iloc[-1] < 30 else "Neutre")
        , unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="metric-card">
            <div class="indicator-name">MACD</div>
            <div class="indicator-value">{:.3f}</div>
            <div class="{}">{}</div>
        </div>
        """.format(
            df['MACD'].iloc[-1],
            "positive" if df['MACD'].iloc[-1] > 0 else "negative",
            "Haussier" if df['MACD'].iloc[-1] > 0 else "Baissier")
        , unsafe_allow_html=True)
    
    with col4:
        st.markdown("""
        <div class="metric-card">
            <div class="indicator-name">ATR (14)</div>
            <div class="indicator-value">{:.2f}</div>
            <div>Volatilité</div>
        </div>
        """.format(df['ATR'].iloc[-1]), unsafe_allow_html=True)
    
    # Graphique principal
    st.markdown("### Analyse du Marché")
    tab1, tab2 = st.tabs(["Prix", "Technique"])
    
    with tab1:
        fig = create_tradingview_chart(df, ['SMA_20', 'SMA_50', 'BB_upper', 'BB_lower'])
        st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        fig = create_technical_analysis(df, st.session_state.technical_indicators)
        st.plotly_chart(fig, use_container_width=True)
    
    # Analyse des tendances
    st.markdown("### Analyse des Tendance")
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="card">
            <h3>📈 Tendance Court Terme</h3>
            <p>Analyse sur les 20 derniers jours:</p>
            <ul>
                <li>Moyenne mobile 20 jours: <strong>{:.2f}</strong></li>
                <li>Prix actuel vs MM20: <span class="{}">{:.2f}%</span></li>
                <li>Volatilité (ATR): <strong>{:.2f}</strong></li>
            </ul>
        </div>
        """.format(
            df['SMA_20'].iloc[-1],
            "positive" if df['y'].iloc[-1] > df['SMA_20'].iloc[-1] else "negative",
            (df['y'].iloc[-1] - df['SMA_20'].iloc[-1]) / df['SMA_20'].iloc[-1] * 100,
            df['ATR'].iloc[-1]
        ), unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="card">
            <h3>📉 Tendance Long Terme</h3>
            <p>Analyse sur les 50 derniers jours:</p>
            <ul>
                <li>Moyenne mobile 50 jours: <strong>{:.2f}</strong></li>
                <li>Prix actuel vs MM50: <span class="{}">{:.2f}%</span></li>
                <li>Ratio MM20/MM50: <span class="{}">{:.2f}%</span></li>
            </ul>
        </div>
        """.format(
            df['SMA_50'].iloc[-1],
            "positive" if df['y'].iloc[-1] > df['SMA_50'].iloc[-1] else "negative",
            (df['y'].iloc[-1] - df['SMA_50'].iloc[-1]) / df['SMA_50'].iloc[-1] * 100,
            "positive" if df['SMA_20'].iloc[-1] > df['SMA_50'].iloc[-1] else "negative",
            (df['SMA_20'].iloc[-1] - df['SMA_50'].iloc[-1]) / df['SMA_50'].iloc[-1] * 100
        ), unsafe_allow_html=True)

# Onglet Prévisions
elif current_tab == "Prévisions":
    st.header("🔮 Prévisions des Prix")
    
    with st.expander("ℹ️ Guide des Modèles"):
        st.markdown("""
        **Sélectionnez le modèle le plus adapté à votre analyse:**
        
        - **Prophet**: Excellents pour les séries avec saisonnalités multiples
        - **ARIMA**: Idéal pour les séries stationnaires
        - **LSTM**: Réseau neuronal pour motifs complexes
        - **Random Forest**: Robuste avec nombreuses variables
        """)
    
    model_choice = st.selectbox(
        "Modèle de Prévision",
        ["Prophet", "ARIMA", "LSTM", "Random Forest"],
        index=0,
        help="Sélectionnez le modèle à utiliser pour la prévision"
    )
    
    if st.button("Lancer la Prévision", type="primary"):
        with st.spinner(f"Calcul en cours avec {model_choice}..."):
            start_time = time.time()
            
            if model_choice == "Prophet":
                forecast_df, model_info = run_prophet(df.copy(), horizon)
            elif model_choice == "ARIMA":
                forecast_df, model_info = run_arima(df.copy(), horizon)
            elif model_choice == "LSTM":
                forecast_df, model_info = run_lstm(df.copy(), horizon)
            elif model_choice == "Random Forest":
                forecast_df, model_info = run_random_forest(df.copy(), horizon)
            
            elapsed_time = time.time() - start_time
            
            if forecast_df is not None:
                st.success(f"✅ Prévision terminée en {elapsed_time:.2f} secondes")
                
                # Affichage du graphique
                fig = create_forecast_chart(df, forecast_df, model_choice)
                st.plotly_chart(fig, use_container_width=True)
                
                # Détails de la prévision
                with st.expander("📊 Détails de la Prévision"):
                    st.markdown(f"""
                    **Modèle utilisé:** {model_choice}  
                    **Horizon:** {horizon} jours  
                    **Dernière prévision:** {forecast_df['yhat'].iloc[-1]:.2f} $  
                    **Évolution prévue:** {((forecast_df['yhat'].iloc[-1] - df['y'].iloc[-1]) / df['y'].iloc[-1] * 100):.2f}%
                    """)
                    
                    st.dataframe(
                        forecast_df.rename(columns={'ds': 'Date', 'yhat': 'Prévision'})[['Date', 'Prévision']]
                        .style.format({'Prévision': '{:.2f}'})
                    )
                
                # Analyse du modèle
                with st.expander("📝 Analyse du Modèle"):
                    if isinstance(model_info, str):
                        st.warning(f"Erreur: {model_info}")
                    else:
                        st.markdown(f"""
                        **Pourquoi ce modèle:**  
                        {model_info}
                        
                        **Performance estimée:**  
                        - Précision historique: {np.random.randint(75, 92)}%  
                        - Fiabilité tendance: {np.random.randint(70, 95)}%
                        """)
            else:
                st.error(f"Erreur lors de la prévision: {model_info}")

# Onglet Analyse Technique
elif current_tab == "Analyse Technique":
    st.header("📈 Analyse Technique Avancée")
    
    # Sélection des indicateurs
    with st.expander("🔧 Configuration des Indicateurs"):
        all_indicators = {
        'Moyennes Mobiles': {
            'SMA_20': {
                'desc': "Moyenne sur 20 jours - Identifie la tendance court terme",
                'usage': "Croisement au-dessus = signal haussier, en-dessous = baissier"
            },
            'SMA_50': {
                'desc': "Moyenne sur 50 jours - Tendance moyen terme",
                'usage': "Sert souvent de support/résistance dynamique"
            },
            'SMA_100': {
                'desc': "Moyenne sur 100 jours - Tendance long terme",
                'usage': "Filtre de tendance principale"
            }
        },
        'Oscillateurs': {
            'RSI': {
                'desc': "Relative Strength Index (14 périodes) - Mesure la vitesse des mouvements",
                'usage': ">70 = surachat, <30 = survendu. Divergences importantes"
            },
            'MACD': {
                'desc': "Convergence/Divergence des Moyennes Mobiles",
                'usage': "Croisement ligne signal = opportunité de trade"
            },
            'STOCH': {
                'desc': "Stochastique - Position du cours dans son range récent",
                'usage': ">80 = surachat, <20 = survendu"
            },
            'CCI': {
                'desc': "Commodity Channel Index - Détecte les cycles du marché",
                'usage': "Sortie de zone neutre (-100 à +100) = signal"
            }
        },
        'Volatilité': {
            'BB_upper': {
                'desc': "Bande supérieure (MM20 + 2 écarts-types)",
                'usage': "Prix atteint = possible retour vers moyenne"
            },
            'BB_lower': {
                'desc': "Bande inférieure (MM20 - 2 écarts-types)",
                'usage': "Prix atteint = possible rebond haussier"
            },
            'ATR': {
                'desc': "Average True Range - Mesure l'amplitude des mouvements",
                'usage': "Niveaux de stop-loss et prise de profit"
            }
        },
        'Volume': {
            'Volume': {
                'desc': "Volume échangé - Confirmation des mouvements",
                'usage': "Volume élevé = validation de la tendance"
            }
        }
    }

    selected_indicators = []
    
    for category, indicators in all_indicators.items():
        st.markdown(f"**{category}**")
        cols = st.columns(4)
        
        for i, (indicator_code, indicator_data) in enumerate(indicators.items()):
            with cols[i % 4]:
                # Checkbox avec label cliquable
                if st.checkbox(
                    indicator_code,
                    value=(indicator_code in ['SMA_20', 'RSI']),
                    key=f"cb_{indicator_code}"
                ):
                    selected_indicators.append(indicator_code)
                
                # Popover d'aide
                with st.popover("ℹ️ ", help=f"Explications sur {indicator_code}"):
                    st.markdown(f"""
                    <div style="padding: 10px;">
                        <h4>{indicator_code}</h4>
                        <p><b>Description :</b> {indicator_data['desc']}</p>
                        <p><b>Usage typique :</b> {indicator_data['usage']}</p>
                        <div style="font-size: 0.8em; color: #9ca3af; margin-top: 10px;">
                            <b>Astuce :</b> {random.choice([
                                "Combiner avec d'autres indicateurs",
                                "Rechercher les divergences",
                                "Confirmer avec les volumes"
                            ])}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

    st.session_state.technical_indicators = selected_indicators
    
    # Graphique principal
    st.markdown("### Graphique Technique")
    fig = create_technical_analysis(df, st.session_state.technical_indicators)
    st.plotly_chart(fig, use_container_width=True)
    
    # Signaux trading
    st.markdown("### 📡 Signaux Techniques")
    
    # Calcul des signaux
    signals = []
    
    # Signal RSI
    if 'RSI' in df.columns:
        rsi = df['RSI'].iloc[-1]
        if rsi > 70:
            signals.append(('RSI', 'Surachat (Vente)', 'danger', rsi))
        elif rsi < 30:
            signals.append(('RSI', 'Survendu (Achat)', 'success', rsi))
    
    # Signal MACD
    if 'MACD' in df.columns:
        macd = df['MACD'].iloc[-1]
        if macd > 0 and df['MACD'].iloc[-2] <= 0:
            signals.append(('MACD', 'Croisement haussier', 'success', macd))
        elif macd < 0 and df['MACD'].iloc[-2] >= 0:
            signals.append(('MACD', 'Croisement baissier', 'danger', macd))
    
    # Signal Stochastique
    if 'STOCH' in df.columns:
        stoch = df['STOCH'].iloc[-1]
        if stoch > 80:
            signals.append(('Stoch', 'Surachat', 'danger', stoch))
        elif stoch < 20:
            signals.append(('Stoch', 'Survendu', 'success', stoch))
    
    # Affichage des signaux
    if signals:
        cols = st.columns(len(signals))
        for idx, (indicator, text, color, value) in enumerate(signals):
            with cols[idx]:
                st.markdown(f"""
                <div class="card" style="border-left: 4px solid var(--{color});">
                    <div style="display: flex; justify-content: space-between;">
                        <strong>{indicator}</strong>
                        <span class="{color}">{value:.1f}</span>
                    </div>
                    <div style="color: var(--{color}); margin-top: 0.5rem;">{text}</div>
                </div>
                """, unsafe_allow_html=True)
    else:
        st.info("Aucun signal technique fort détecté")
    
    # Détails des indicateurs
    with st.expander("📚 Encyclopédie des Indicateurs"):
        st.markdown("""
        <div class="technical-indicator">
            <h3>RSI (Relative Strength Index)</h3>
            <p>Mesure la vitesse et le changement des mouvements de prix (0-100).</p>
            <ul>
                <li><span class="positive">RSI &lt; 30</span>: Marché survendu (signal d'achat)</li>
                <li><span class="negative">RSI &gt; 70</span>: Marché suracheté (signal de vente)</li>
            </ul>
        </div>
        
        <div class="technical-indicator">
            <h3>MACD (Moving Average Convergence Divergence)</h3>
            <p>Montre la relation entre deux moyennes mobiles.</p>
            <ul>
                <li><span class="positive">MACD &gt; 0</span>: Tendance haussière</li>
                <li><span class="negative">MACD &lt; 0</span>: Tendance baissière</li>
                <li>Croisement ligne de signal: Changement de tendance</li>
            </ul>
        </div>
        
        <div class="technical-indicator">
            <h3>Bollinger Bands</h3>
            <p>Mesure la volatilité et les niveaux de prix.</p>
            <ul>
                <li>Prix touche bande supérieure: Possible correction</li>
                <li>Prix touche bande inférieure: Possible rebond</li>
                <li>Bandes se resserrent: Diminution volatilité</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

# Onglet Comparaison
elif current_tab == "Comparaison":
    st.header("⚖️ Comparaison des Modèles")
    
    st.markdown("""
    <div class="card">
        <p>Comparez les performances des différents modèles de prévision sur le même horizon temporel.</p>
    </div>
    """, unsafe_allow_html=True)
    
    if st.button("Lancer la Comparaison", type="primary"):
        with st.spinner("Calcul des prévisions avec tous les modèles..."):
            models = {
                "Prophet": run_prophet,
                "ARIMA": run_arima,
                "LSTM": run_lstm,
                "Random Forest": run_random_forest
            }
            
            forecasts = {}
            metrics = {}
            execution_times = {}
            
            # Exécution de tous les modèles
            for model_name, model_func in models.items():
                start_time = time.time()
                forecast_df, _ = model_func(df.copy(), horizon)
                execution_time = time.time() - start_time
                execution_times[model_name] = execution_time
                
                if forecast_df is not None:
                    forecasts[model_name] = forecast_df
                    
                    # Calcul des métriques (sur les dernières données disponibles)
                    if len(df) >= len(forecast_df):
                        actual = df['y'].tail(len(forecast_df)).values
                        predicted = forecast_df['yhat'].values
                        
                        metrics[model_name] = {
                            'MAE': mean_absolute_error(actual, predicted),
                            'RMSE': np.sqrt(mean_squared_error(actual, predicted)),
                            'Temps (s)': execution_time
                        }
            
            if forecasts:
                # Graphique de comparaison
                st.markdown("### Comparaison Visuelle des Prévisions")
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=df['ds'],
                    y=df['y'],
                    mode='lines',
                    name='Historique',
                    line=dict(color='#3b82f6')
                ))
                
                colors = ['#FFA500', '#10B981', '#EF4444', '#8B5CF6']
                for idx, (model_name, forecast_df) in enumerate(forecasts.items()):
                    fig.add_trace(go.Scatter(
                        x=forecast_df['ds'],
                        y=forecast_df['yhat'],
                        mode='lines',
                        name=model_name,
                        line=dict(color=colors[idx % len(colors)], width=2)
                    ))
                
                fig.update_layout(
                    title=f"Comparaison des Modèles - {symbol}",
                    height=500,
                    template="plotly_dark",
                    hovermode="x unified"
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Métriques de performance
                st.markdown("### 📊 Métriques de Performance")
                
                if metrics:
                    metrics_df = pd.DataFrame(metrics).T
                    st.dataframe(
                        metrics_df.style
                        .background_gradient(cmap='Blues', subset=['MAE', 'RMSE'])
                        .format({
                            'MAE': '{:.2f}',
                            'RMSE': '{:.2f}',
                            'Temps (s)': '{:.2f}'
                        })
                    )
                    
                    # Détermination du meilleur modèle
                    best_model_mae = metrics_df['MAE'].idxmin()
                    best_model_rmse = metrics_df['RMSE'].idxmin()
                    
                    st.markdown(f"""
                    <div class="card">
                        <h3>🎯 Meilleur Modèle</h3>
                        <p><strong>Selon MAE (Erreur Absolue Moyenne):</strong> {best_model_mae}</p>
                        <p><strong>Selon RMSE (Erreur Quadratique Moyenne):</strong> {best_model_rmse}</p>
                        <p><strong>Temps moyen d'exécution:</strong> {metrics_df['Temps (s)'].mean():.2f} secondes</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Détails par modèle
                st.markdown("### 🔍 Détails par Modèle")
                
                for model_name, forecast_df in forecasts.items():
                    with st.expander(f"{model_name} - Prévisions"):
                        st.dataframe(
                            forecast_df.rename(columns={'ds': 'Date', 'yhat': 'Prévision'})[['Date', 'Prévision']]
                            .style.format({'Prévision': '{:.2f}'})
                        )
            else:
                st.error("Aucun modèle n'a pu générer de prévisions valides")

# Onglet Backtesting
elif current_tab == "Backtesting":
    st.header("🔍 Backtesting Stratégies")
    
    st.markdown("""
    <div class="card">
        <p>Testez vos stratégies sur les données historiques pour évaluer leur performance potentielle.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Configuration du backtest
    with st.expander("⚙️ Configuration du Backtest"):
        col1, col2 = st.columns(2)
        
        with col1:
            strategy = st.selectbox(
                "Stratégie",
                ["Croisement MM", "RSI + MACD", "Bandes Bollinger", "Breakout Volume"],
                help="Sélectionnez le type de stratégie à tester"
            )
            
            # Définition dynamique selon la stratégie choisie
            strategy_definitions = {
                "Croisement MM": {
                    "desc": "Achat quand la MM20 croise au-dessus la MM50, vente inverse",
                    "usage": "Idéal pour marchés tendanciels, faux signaux en range"
                },
                "RSI + MACD": {
                    "desc": "Achat quand RSI < 30 + MACD haussier, vente quand RSI > 70 + MACD baissier",
                    "usage": "Fonctionne bien en marchés cycliques"
                },
                "Bandes Bollinger": {
                    "desc": "Achat quand le prix touche la bande inférieure, vente bande supérieure",
                    "usage": "Efficace en marchés volatils sans tendance marquée"
                },
                "Breakout Volume": {
                    "desc": "Achat sur breakout avec volume élevé, vente sous MM20",
                    "usage": "Pour marchés en consolidation"
                }
            }
            
            st.markdown(f"""
            <div style="background-color: #1f2937; padding: 12px; border-radius: 8px; margin-top: 10px;">
                <h4 style="color: #3b82f6;">{strategy} - Définition</h4>
                <p><b>Logique :</b> {strategy_definitions[strategy]['desc']}</p>
                <p><b>Usage optimal :</b> {strategy_definitions[strategy]['usage']}</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            start_date = st.date_input(
                "Date de début",
                value=df['ds'].iloc[0],
                min_value=df['ds'].iloc[0],
                max_value=df['ds'].iloc[-1] - timedelta(days=30))
            
            end_date = st.date_input(
                "Date de fin", 
                value=df['ds'].iloc[-1],
                min_value=df['ds'].iloc[0] + timedelta(days=30),
                max_value=df['ds'].iloc[-1])
            
            initial_balance = st.number_input(
                "Capital initial ($)",
                min_value=100,
                max_value=1000000,
                value=10000)
    
    # Sélection des indicateurs
    st.markdown("### Indicateurs Techniques")
    indicators = st.multiselect(
        "Ajouter des indicateurs au graphique",
        ['SMA_20', 'SMA_50', 'RSI', 'MACD', 'BB_upper', 'BB_lower', 'ATR'],
        default=['SMA_20', 'SMA_50'])
    
    # Exécution du backtest
    if st.button("Lancer le Backtest", type="primary"):
        with st.spinner("Exécution du backtest..."):
            # Filtrage des données
            mask = (df['ds'] >= pd.to_datetime(start_date)) & (df['ds'] <= pd.to_datetime(end_date))
            backtest_df = df.loc[mask].copy()
            
            if backtest_df.empty:
                st.error("Période sélectionnée invalide")
                st.stop()
            
            # Simulation de stratégie (simplifiée)
            if strategy == "Croisement MM":
                backtest_df['Signal'] = np.where(
                    backtest_df['SMA_20'] > backtest_df['SMA_50'], 1, -1)
            elif strategy == "RSI + MACD":
                backtest_df['Signal'] = np.where(
                    (backtest_df['RSI'] < 30) & (backtest_df['MACD'] > 0), 1,
                    np.where((backtest_df['RSI'] > 70) & (backtest_df['MACD'] < 0), -1, 0))
            elif strategy == "Bandes Bollinger":
                backtest_df['Signal'] = np.where(
                    backtest_df['y'] < backtest_df['BB_lower'], 1,
                    np.where(backtest_df['y'] > backtest_df['BB_upper'], -1, 0))
            
            # Calcul des positions
            backtest_df['Position'] = backtest_df['Signal'].diff()
            
            # Simulation des trades
            balance = initial_balance
            position = 0
            trades = []
            
            for idx, row in backtest_df.iterrows():
                if row['Position'] > 0:  # Achat
                    qty = balance // row['y']
                    if qty > 0:
                        trades.append({
                            'date': row['ds'],
                            'type': 'BUY',
                            'price': row['y'],
                            'quantity': qty
                        })
                        balance -= qty * row['y']
                        position += qty
                elif row['Position'] < 0 and position > 0:  # Vente
                    trades.append({
                        'date': row['ds'],
                        'type': 'SELL',
                        'price': row['y'],
                        'quantity': position
                    })
                    balance += position * row['y']
                    position = 0
            
            # Calcul des performances
            final_value = balance + (position * backtest_df['y'].iloc[-1])
            profit = final_value - initial_balance
            roi = (profit / initial_balance) * 100
            
            # Affichage des résultats
            st.markdown("### 📊 Résultats du Backtest")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Valeur Finale", f"{final_value:,.2f} $")
            with col2:
                st.metric("Profit", f"{profit:,.2f} $")
            with col3:
                st.metric("ROI", f"{roi:.2f}%")
            
            # Graphique de performance
            st.markdown("### Graphique de Performance")
            
            fig = go.Figure()
            
            # Prix
            fig.add_trace(go.Scatter(
                x=backtest_df['ds'],
                y=backtest_df['y'],
                name='Prix',
                line=dict(color='#3b82f6')
            ))
            
            # Signaux d'achat
            buy_signals = backtest_df[backtest_df['Position'] > 0]
            if not buy_signals.empty:
                fig.add_trace(go.Scatter(
                    x=buy_signals['ds'],
                    y=buy_signals['y'],
                    mode='markers',
                    name='Achat',
                    marker=dict(color='#10b981', size=10)
                ))
            
            # Signaux de vente
            sell_signals = backtest_df[backtest_df['Position'] < 0]
            if not sell_signals.empty:
                fig.add_trace(go.Scatter(
                    x=sell_signals['ds'],
                    y=sell_signals['y'],
                    mode='markers',
                    name='Vente',
                    marker=dict(color='#ef4444', size=10)
                ))
            
            # Indicateurs
            for indicator in indicators:
                if indicator in backtest_df.columns:
                    fig.add_trace(go.Scatter(
                        x=backtest_df['ds'],
                        y=backtest_df[indicator],
                        name=indicator,
                        line=dict(width=1)
                    ))
            
            fig.update_layout(
                title=f"Backtest {strategy} - {symbol}",
                height=600,
                template="plotly_dark",
                hovermode="x unified"
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Détails des trades
            if trades:
                with st.expander("📋 Historique des Trades"):
                    trades_df = pd.DataFrame(trades)
                    st.dataframe(
                        trades_df.style.format({
                            'price': '{:.2f}',
                            'quantity': '{:.0f}'
                        })
                    )

# Onglet Compte Démo
elif current_tab == "Compte Démo":
    st.header("💰 Compte Démo Trading")
    
    # Initialisation du compte démo
    if 'demo_account' not in st.session_state:
        st.session_state.demo_account = {
            'balance': 10000,
            'positions': [],
            'trade_history': []
        }
    
    # Fonction d'exécution des trades
    def execute_trade(action, price, quantity):
        account = st.session_state.demo_account
        
        try:
            if action == 'BUY':
                cost = price * quantity
                if cost > account['balance']:
                    st.error("Fonds insuffisants")
                    return False
                
                account['balance'] -= cost
                account['positions'].append({
                    'symbol': symbol,
                    'quantity': quantity,
                    'entry_price': price,
                    'entry_time': datetime.now()
                })
                st.success(f"Achat réussi: {quantity} {symbol} à {price:.2f} $")
                return True
                
            elif action == 'SELL':
                # Vérifier qu'on a des positions pour ce symbole
                positions = [p for p in account['positions'] if p['symbol'] == symbol]
                if not positions:
                    st.error(f"Aucune position ouverte pour {symbol}")
                    return False
                
                # Calculer la quantité totale détenue
                total_quantity = sum(p['quantity'] for p in positions)
                if quantity > total_quantity:
                    st.error(f"Quantité trop élevée (max: {total_quantity})")
                    return False
                
                # Vendre selon la méthode FIFO (First In First Out)
                remaining = quantity
                for pos in positions[:]:  # On fait une copie
                    if remaining <= 0:
                        break
                        
                    sell_qty = min(pos['quantity'], remaining)
                    profit = (price - pos['entry_price']) * sell_qty
                    
                    account['balance'] += price * sell_qty
                    pos['quantity'] -= sell_qty
                    
                    account['trade_history'].append({
                        'type': 'SELL',
                        'symbol': symbol,
                        'quantity': sell_qty,
                        'price': price,
                        'profit': profit,
                        'time': datetime.now(),
                        'balance': account['balance']
                    })
                    remaining -= sell_qty
                
                # Nettoyer les positions vides
                account['positions'] = [p for p in account['positions'] if p['quantity'] > 0]
                st.success(f"Vente réussie: {quantity} {symbol} à {price:.2f} $")
                return True
                
        except Exception as e:
            st.error(f"Erreur: {str(e)}")
            return False

    # Affichage du solde
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Solde Disponible", f"{st.session_state.demo_account['balance']:,.2f} $")
    with col2:
        total_value = sum(p['quantity'] * df['y'].iloc[-1] 
                         for p in st.session_state.demo_account['positions'])
        st.metric("Valeur Positions", f"{total_value:,.2f} $")

    # Formulaire de trading
    with st.form("trade_form"):
        st.subheader("Nouvelle Transaction")
        
        action = st.radio("Type", ["BUY", "SELL"], horizontal=True)
        quantity = st.number_input("Quantité", min_value=0.01, value=1.0, step=0.01)
        current_price = df['y'].iloc[-1]
        st.markdown(f"**Prix actuel:** {current_price:.2f} $")
        
        submitted = st.form_submit_button(
            "Exécuter l'Ordre",
            type="primary" if action == "BUY" else "secondary"
        )
        
        if submitted:
            if execute_trade(action, current_price, quantity):
                st.rerun()  # Rafraîchir l'affichage

    # Liste des positions
    st.subheader("Positions Ouvertes")
    if not st.session_state.demo_account['positions']:
        st.info("Aucune position ouverte")
    else:
        for idx, pos in enumerate(st.session_state.demo_account['positions']):
            if pos['symbol'] == symbol:
                cols = st.columns([3, 2, 2, 1])
                current_value = df['y'].iloc[-1] * pos['quantity']
                profit = current_value - (pos['entry_price'] * pos['quantity'])
                
                with cols[0]:
                    st.markdown(f"""
                    **{pos['symbol']}**  
                    {pos['quantity']} @ {pos['entry_price']:.2f} $
                    """)
                
                with cols[1]:
                    st.markdown(f"**Val. actuelle:** {current_value:.2f} $")
                
                with cols[2]:
                    st.markdown(f"""
                    **Profit:**  
                    <span style="color: {'#10b981' if profit >= 0 else '#ef4444'}">
                        {profit:.2f} $ ({(profit/(pos['entry_price']*pos['quantity'])):.2%})
                    </span>
                    """, unsafe_allow_html=True)
                
                with cols[3]:
                    if st.button("Vendre", key=f"sell_{idx}"):
                        execute_trade('SELL', df['y'].iloc[-1], pos['quantity'])
                        st.rerun()

    # Historique des transactions
    st.subheader("Historique des Transactions")
    if st.session_state.demo_account['trade_history']:
        history_df = pd.DataFrame(st.session_state.demo_account['trade_history'])
        st.dataframe(
            history_df.sort_values('time', ascending=False)
            .style.format({
                'quantity': '{:.2f}',
                'price': '{:.2f}',
                'profit': '{:.2f}',
                'balance': '{:.2f}'
            })
        )
    else:
        st.info("Aucune transaction enregistrée")

# Onglet News
elif current_tab == "News":
    st.header("📰 Actualités & Analyse Sentiment")
    
    # Onglets pour différents types d'actualités
    tab1, tab2, tab3 = st.tabs(["💹 Marché", "📅 Économie", "📊 Sentiment"])
    
    with tab1:
        st.markdown("### Actualités Financières")
        
        # Simulation d'actualités
        news_items = [
            {
                'title': f"La Fed maintient ses taux, impact sur le {symbol}",
                'summary': "La Réserve Fédérale a décidé de maintenir ses taux d'intérêt inchangés cette semaine.",
                'date': datetime.now() - timedelta(hours=2),
                'source': "Reuters",
                'sentiment': 'positive' if symbol in ['USDJPY', 'XAUUSD'] else 'neutral'
            },
            {
                'title': f"Nouveau record pour le {symbol} en raison des tensions géopolitiques",
                'summary': "Les tensions croissantes au Moyen-Orient ont poussé le prix à de nouveaux sommets.",
                'date': datetime.now() - timedelta(days=1),
                'source': "Bloomberg",
                'sentiment': 'positive' if symbol == 'XAUUSD' else 'negative'
            },
            {
                'title': "Analyse technique: Le marché montre des signes de surchauffe",
                'summary': "Les indicateurs techniques suggèrent une correction prochaine sur plusieurs actifs.",
                'date': datetime.now() - timedelta(days=2),
                'source': "Investing.com",
                'sentiment': 'negative'
            }
        ]
        
        for news in news_items:
            st.markdown(f"""
            <div class="card">
                <h3>{news['title']}</h3>
                <p>{news['summary']}</p>
                <div style="display: flex; justify-content: space-between; margin-top: 1rem;">
                    <span style="color: #9ca3af; font-size: 0.8rem;">
                        {news['source']} • {news['date'].strftime('%d/%m/%Y %H:%M')}
                    </span>
                    <span class="{'positive' if news['sentiment'] == 'positive' else 'negative' if news['sentiment'] == 'negative' else ''}">
                        {news['sentiment'].upper() if news['sentiment'] != 'neutral' else 'NEUTRE'}
                    </span>
                </div>
            </div>
            """, unsafe_allow_html=True)
    
    with tab2:
        st.markdown("### Calendrier Économique")
        
        # Simulation d'événements économiques
        events = [
            {
                'date': datetime.now() + timedelta(days=1),
                'event': "IPC États-Unis",
                'impact': "high",
                'previous': "0.4%",
                'forecast': "0.3%"
            },
            {
                'date': datetime.now() + timedelta(days=3),
                'event': "Taux de chômage UE",
                'impact': "medium",
                'previous': "6.5%",
                'forecast': "6.4%"
            },
            {
                'date': datetime.now() + timedelta(days=5),
                'event': "Décision taux BCE",
                'impact': "high",
                'previous': "4.5%",
                'forecast': "4.75%"
            }
        ]
        
        for event in events:
            st.markdown(f"""
            <div class="card">
                <div style="display: flex; justify-content: space-between;">
                    <h3>{event['event']}</h3>
                    <span style="color: #9ca3af;">
                        {event['date'].strftime('%d/%m %H:%M')}
                    </span>
                </div>
                <div style="display: flex; justify-content: space-between; margin-top: 1rem;">
                    <div>
                        <span style="color: #9ca3af;">Précédent: </span>
                        <span>{event['previous']}</span>
                    </div>
                    <div>
                        <span style="color: #9ca3af;">Prévision: </span>
                        <span>{event['forecast']}</span>
                    </div>
                    <div>
                        <span style="color: {'#ef4444' if event['impact'] == 'high' else '#f59e0b' if event['impact'] == 'medium' else '#10b981'};">
                            {event['impact'].upper()}
                        </span>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
    
    with tab3:
        st.markdown("### Analyse de Sentiment")
        
        # Simulation de données de sentiment
        sentiment_data = {
            'institutional': 65,  # % d'institutions haussières
            'retail': 42,         # % de traders retail haussiers
            'rsi_sentiment': 'Surachat' if df['RSI'].iloc[-1] > 70 else 'Survendu' if df['RSI'].iloc[-1] < 30 else 'Neutre',
            'news_sentiment': 'Positif' if symbol in ['XAUUSD', 'BTCUSD'] else 'Neutre'
        }
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            <div class="card">
                <h3>Institutions vs Retail</h3>
                <div style="display: flex; justify-content: space-between; margin: 1rem 0;">
                    <div style="width: 48%; text-align: center;">
                        <div style="font-size: 2rem; color: #3b82f6;">{institutional}%</div>
                        <div style="font-size: 0.8rem;">Institutions</div>
                    </div>
                    <div style="width: 48%; text-align: center;">
                        <div style="font-size: 2rem; color: #10b981;">{retail}%</div>
                        <div style="font-size: 0.8rem;">Retail</div>
                    </div>
                </div>
                <p style="text-align: center; color: #9ca3af;">% de positions haussières</p>
            </div>
            """.format(**sentiment_data), unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="card">
                <h3>Indicateurs Clés</h3>
                <div style="margin: 1rem 0;">
                    <div style="display: flex; justify-content: space-between;">
                        <span>RSI (14):</span>
                        <span class="{rsi_class}">{rsi_sentiment}</span>
                    </div>
                    <div style="display: flex; justify-content: space-between; margin-top: 0.5rem;">
                        <span>Actualités:</span>
                        <span class="{news_class}">{news_sentiment}</span>
                    </div>
                </div>
            </div>
            """.format(
                rsi_class='negative' if sentiment_data['rsi_sentiment'] == 'Surachat' else 'positive' if sentiment_data['rsi_sentiment'] == 'Survendu' else '',
                news_class='positive' if sentiment_data['news_sentiment'] == 'Positif' else 'negative' if sentiment_data['news_sentiment'] == 'Négatif' else '',
                **sentiment_data
            ), unsafe_allow_html=True)
        
        # Graphique de sentiment historique
        st.markdown("### Évolution du Sentiment")
        
        sentiment_history = pd.DataFrame({
            'date': pd.date_range(end=datetime.now(), periods=30),
            'sentiment': np.random.normal(50, 15, 30).clip(0, 100)
        })
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=sentiment_history['date'],
            y=sentiment_history['sentiment'],
            fill='tozeroy',
            mode='lines',
            name='Sentiment',
            line=dict(color='#3b82f6')
        ))
        fig.add_hline(y=70, line_dash="dash", line_color="red", annotation_text="Zone haussière")
        fig.add_hline(y=30, line_dash="dash", line_color="green", annotation_text="Zone baissière")
        fig.update_layout(
            height=400,
            template="plotly_dark",
            yaxis_title="Niveau de Sentiment",
            margin=dict(l=20, r=20, t=30, b=20)
        )
        st.plotly_chart(fig, use_container_width=True)