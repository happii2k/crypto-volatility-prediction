
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import joblib
import warnings
warnings.filterwarnings("ignore")

# Import the predictor class
from crypto_volatility_trainer import CryptocurrencyVolatilityPredictor

# Page configuration
st.set_page_config(
    page_title="Crypto Volatility Predictor",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #ff7f0e;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
</style>
""", unsafe_allow_html=True)

class CryptoVolatilityApp:
    def __init__(self):
        self.predictor = None
        self.sample_data = None

    @st.cache_resource
    def load_model(_self):
        """Load the trained model"""
        try:
            predictor = CryptocurrencyVolatilityPredictor.load_model('crypto_volatility_model.pkl')
            return predictor
        except FileNotFoundError:
            return None

    @st.cache_data
    def load_sample_data(_self):
        """Load sample data for demonstration"""
        try:
            df = pd.read_csv('dataset.csv')
            return df
        except FileNotFoundError:
            # Create sample data if original not available
            return _self.create_sample_data()

    def create_sample_data(self):
        """Create sample data for demonstration"""
        np.random.seed(42)

        cryptos = ['Bitcoin', 'Ethereum', 'Litecoin', 'XRP', 'Cardano']
        dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')

        data = []
        for crypto in cryptos:
            for date in dates:
                data.append({
                    'crypto_name': crypto,
                    'date': date,
                    'open': np.random.uniform(100, 50000),
                    'high': np.random.uniform(100, 55000),
                    'low': np.random.uniform(50, 48000),
                    'close': np.random.uniform(100, 52000),
                    'volume': np.random.uniform(1e6, 1e10),
                    'marketCap': np.random.uniform(1e9, 1e12)
                })

        return pd.DataFrame(data)

    def prepare_input_data(self, crypto_name, open_price, high_price, low_price, 
                          close_price, volume, market_cap, selected_date):
        """Prepare input data for prediction"""
        # Create a single row dataframe
        data = {
            'crypto_name': [crypto_name],
            'date': [selected_date],
            'open': [open_price],
            'high': [high_price],
            'low': [low_price],
            'close': [close_price],
            'volume': [volume],
            'marketCap': [market_cap]
        }

        df = pd.DataFrame(data)
        df['date'] = pd.to_datetime(df['date'])

        # Engineer features (same as training)
        df['price_change'] = df['close'] - df['open']
        df['range'] = df['high'] - df['low']
        df['return_pct'] = (df['close'] - df['open']) / df['open'] * 100
        df['volume_change'] = 0.0  # For single prediction
        df['marketCap_change'] = 0.0  # For single prediction
        df['day_of_week'] = df['date'].apply(lambda x: x.weekday())
        df['quarter'] = df['date'].apply(lambda x: x.quarter)
        df['volatility_14d'] = 5.0  # Default value for demo
        df['gk_volatility'] = np.sqrt(
            0.5 * (np.log(df['high'] / df['low']))**2 - 
            (2*np.log(2)-1) * (np.log(df['close'] / df['open']))**2
        )

        # Encode crypto name
        if self.predictor and hasattr(self.predictor.label_encoder, 'classes_'):
            try:
                df['crypto_encoded'] = self.predictor.label_encoder.transform([crypto_name])[0]
            except ValueError:
                # If crypto not in training data, use default
                df['crypto_encoded'] = 0
        else:
            df['crypto_encoded'] = 0

        return df

    def run(self):
        """Main app function"""
        # Header
        st.markdown('<h1 class="main-header">🚀 Cryptocurrency Volatility Predictor</h1>', 
                   unsafe_allow_html=True)

        st.markdown("""
        <div style="text-align: center; margin-bottom: 2rem;">
            <p style="font-size: 1.2rem; color: #666;">
                Predict cryptocurrency volatility using advanced machine learning models
            </p>
        </div>
        """, unsafe_allow_html=True)

        # Load model and data
        self.predictor = self.load_model()
        self.sample_data = self.load_sample_data()

        if self.predictor is None:
            st.error("""
            ⚠️ **Model not found!** 

            Please train the model first by running:
            ```python
            python crypto_volatility_trainer.py
            ```

            For demonstration purposes, you can still use the interface below.
            """)

        # Sidebar for inputs
        with st.sidebar:
            st.markdown('<h2 class="sub-header">📊 Input Parameters</h2>', unsafe_allow_html=True)

            # Cryptocurrency selection
            if self.sample_data is not None:
                available_cryptos = self.sample_data['crypto_name'].unique().tolist()
            else:
                available_cryptos = ['Bitcoin', 'Ethereum', 'Litecoin', 'XRP', 'Cardano']

            crypto_name = st.selectbox(
                "Select Cryptocurrency",
                available_cryptos,
                index=0
            )

            # Date selection
            selected_date = st.date_input(
                "Select Date",
                value=datetime.now().date(),
                min_value=datetime(2020, 1, 1).date(),
                max_value=datetime.now().date()
            )

            # Price inputs
            st.markdown("### 💰 Price Data")
            col1, col2 = st.columns(2)

            with col1:
                open_price = st.number_input("Open Price ($)", min_value=0.01, value=45000.0, step=100.0)
                high_price = st.number_input("High Price ($)", min_value=0.01, value=46000.0, step=100.0)

            with col2:
                low_price = st.number_input("Low Price ($)", min_value=0.01, value=44000.0, step=100.0)
                close_price = st.number_input("Close Price ($)", min_value=0.01, value=45500.0, step=100.0)

            # Volume and Market Cap
            st.markdown("### 📈 Market Data")
            volume = st.number_input("Volume", min_value=1, value=1000000000, step=1000000)
            market_cap = st.number_input("Market Cap", min_value=1, value=800000000000, step=1000000000)

            # Prediction button
            predict_button = st.button("🔮 Predict Volatility", use_container_width=True)

        # Main content area
        col1, col2 = st.columns([2, 1])

        with col1:
            if predict_button:
                # Validate inputs
                if high_price < max(open_price, close_price) or low_price > min(open_price, close_price):
                    st.error("⚠️ Please ensure High ≥ max(Open, Close) and Low ≤ min(Open, Close)")
                else:
                    # Prepare input data
                    input_data = self.prepare_input_data(
                        crypto_name, open_price, high_price, low_price, 
                        close_price, volume, market_cap, selected_date
                    )

                    if self.predictor:
                        try:
                            # Make prediction
                            prediction = self.predictor.predict(input_data)
                            volatility = prediction[0]

                            # Display prediction result
                            st.markdown('<h2 class="sub-header">🎯 Prediction Result</h2>', 
                                      unsafe_allow_html=True)

                            # Create metrics
                            col_a, col_b, col_c = st.columns(3)

                            with col_a:
                                st.metric(
                                    "Predicted Volatility", 
                                    f"{volatility:.4f}",
                                    help="30-day rolling volatility prediction"
                                )

                            with col_b:
                                risk_level = "High" if volatility > 5 else "Medium" if volatility > 2 else "Low"
                                risk_color = "🔴" if volatility > 5 else "🟡" if volatility > 2 else "🟢"
                                st.metric("Risk Level", f"{risk_color} {risk_level}")

                            with col_c:
                                price_range = ((high_price - low_price) / open_price) * 100
                                st.metric("Daily Range %", f"{price_range:.2f}%")

                            # Visualization
                            self.create_visualizations(volatility, crypto_name, input_data)

                        except Exception as e:
                            st.error(f"Prediction error: {str(e)}")
                            st.info("Using demo prediction...")
                            demo_volatility = np.random.uniform(1, 8)
                            st.metric("Demo Volatility", f"{demo_volatility:.4f}")
                    else:
                        # Demo prediction
                        demo_volatility = np.random.uniform(1, 8)
                        st.markdown('<h2 class="sub-header">🎯 Demo Prediction</h2>', 
                                  unsafe_allow_html=True)
                        st.metric("Demo Volatility", f"{demo_volatility:.4f}")
                        st.info("This is a demo prediction. Train the model for actual predictions.")

        with col2:
            # Model information
            st.markdown('<h3 class="sub-header">ℹ️ Model Info</h3>', unsafe_allow_html=True)

            if self.predictor and hasattr(self.predictor, 'results'):
                # Display model performance
                best_model = max(self.predictor.results.keys(), 
                               key=lambda k: self.predictor.results[k]['Test R²'])

                st.markdown(f"""
                **Best Model:** {best_model}

                **Performance Metrics:**
                - R² Score: {self.predictor.results[best_model]['Test R²']:.4f}
                - RMSE: {self.predictor.results[best_model]['Test RMSE']:.4f}
                - MAE: {self.predictor.results[best_model]['Test MAE']:.4f}
                """)
            else:
                st.info("""
                **Model Status:** Not loaded

                **Features Used:**
                - Price data (OHLC)
                - Volume and Market Cap
                - Technical indicators
                - Temporal features
                - Historical volatility
                """)

            # About section
            st.markdown('<h3 class="sub-header">📚 About</h3>', unsafe_allow_html=True)
            st.markdown("""
            This application predicts cryptocurrency volatility using:

            - **Machine Learning Models**: XGBoost, Random Forest, etc.
            - **Technical Indicators**: Garman-Klass volatility, price ranges
            - **Market Features**: Volume, market cap changes
            - **Temporal Features**: Day of week, quarter

            **Volatility Interpretation:**
            - **Low (0-2)**: Stable price movement
            - **Medium (2-5)**: Moderate price swings  
            - **High (5+)**: Highly volatile, risky
            """)

    def create_visualizations(self, volatility, crypto_name, input_data):
        """Create visualizations for the prediction"""
        st.markdown('<h3 class="sub-header">📊 Visualizations</h3>', unsafe_allow_html=True)

        # Volatility gauge
        fig_gauge = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = volatility,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "Predicted Volatility"},
            gauge = {
                'axis': {'range': [None, 10]},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [0, 2], 'color': "lightgreen"},
                    {'range': [2, 5], 'color': "yellow"},
                    {'range': [5, 10], 'color': "lightcoral"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 7
                }
            }
        ))
        fig_gauge.update_layout(height=300)
        st.plotly_chart(fig_gauge, use_container_width=True)

        # Historical context (if sample data available)
        if self.sample_data is not None:
            crypto_data = self.sample_data[self.sample_data['crypto_name'] == crypto_name]
            if len(crypto_data) > 0:
                # Calculate historical volatility for comparison
                crypto_data = crypto_data.copy()
                crypto_data['return_pct'] = ((crypto_data['close'] - crypto_data['open']) / 
                                           crypto_data['open'] * 100)

                if len(crypto_data) >= 30:
                    crypto_data['volatility'] = crypto_data['return_pct'].rolling(window=30).std()

                    fig_hist = px.line(
                        crypto_data.dropna(), 
                        x='date', 
                        y='volatility',
                        title=f'{crypto_name} Historical Volatility (30-day rolling)'
                    )

                    # Add prediction line
                    fig_hist.add_hline(
                        y=volatility, 
                        line_dash="dash", 
                        line_color="red",
                        annotation_text="Predicted"
                    )

                    st.plotly_chart(fig_hist, use_container_width=True)

# Run the app
if __name__ == "__main__":
    app = CryptoVolatilityApp()
    app.run()
