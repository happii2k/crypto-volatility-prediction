
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

# Import ML libraries
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn.svm import SVR
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.impute import SimpleImputer
import joblib

class CryptocurrencyVolatilityPredictor:
    """
    Cryptocurrency Volatility Prediction Model

    This class handles the complete ML pipeline for predicting cryptocurrency volatility
    including data preprocessing, feature engineering, model training, and evaluation.
    """

    def __init__(self):
        self.scaler = StandardScaler()
        self.imputer = SimpleImputer(strategy='median')
        self.label_encoder = LabelEncoder()
        self.model = None
        self.feature_columns = None
        self.results = {}

    def load_data(self, filepath):
        """Load and perform initial data cleaning"""
        df = pd.read_csv(filepath)

        # Drop unnecessary columns
        if 'Unnamed: 0' in df.columns:
            df.drop(columns=['Unnamed: 0'], inplace=True)
        if 'timestamp' in df.columns:
            df.drop(columns=['timestamp'], inplace=True)

        # Convert date to datetime
        df['date'] = pd.to_datetime(df['date'])

        return df

    def engineer_features(self, df):
        """Perform comprehensive feature engineering"""
        df_processed = df.copy()

        # Price-based features
        df_processed['price_change'] = df_processed['close'] - df_processed['open']
        df_processed['range'] = df_processed['high'] - df_processed['low']
        df_processed['return_pct'] = (df_processed['close'] - df_processed['open']) / df_processed['open'] * 100

        # Volume and market cap changes
        df_processed['volume_change'] = df_processed['volume'].pct_change()
        df_processed['marketCap_change'] = df_processed['marketCap'].pct_change()

        # Temporal features
        df_processed['day_of_week'] = df_processed['date'].apply(lambda x: x.weekday())
        df_processed['month'] = df_processed['date'].apply(lambda x: x.month)
        df_processed['quarter'] = df_processed['date'].apply(lambda x: x.quarter)
        df_processed['date_ordinal'] = df_processed['date'].apply(lambda x: x.toordinal())

        # Volatility features (target and predictive)
        df_processed['volatility_7d'] = df_processed['return_pct'].rolling(window=7).std()
        df_processed['volatility_14d'] = df_processed['return_pct'].rolling(window=14).std()
        df_processed['volatility_30d'] = df_processed['return_pct'].rolling(window=30).std()

        # Target variable: Future volatility
        df_processed['target_volatility'] = df_processed['volatility_30d'].shift(-1)

        # Garman-Klass volatility estimator
        df_processed['gk_volatility'] = np.sqrt(
            0.5 * (np.log(df_processed['high'] / df_processed['low']))**2 - 
            (2*np.log(2)-1) * (np.log(df_processed['close'] / df_processed['open']))**2
        )

        return df_processed

    def handle_outliers(self, df, features):
        """Handle outliers using IQR method"""
        df_clean = df.copy()

        for col in features:
            if col in df_clean.columns:
                percentile25 = df_clean[col].quantile(0.25)
                percentile75 = df_clean[col].quantile(0.75)
                iqr = percentile75 - percentile25
                upper_limit = percentile75 + 1.5 * iqr
                lower_limit = percentile25 - 1.5 * iqr

                df_clean.loc[(df_clean[col] > upper_limit), col] = upper_limit
                df_clean.loc[(df_clean[col] < lower_limit), col] = lower_limit

        return df_clean

    def remove_highly_correlated_features(self, df, threshold=0.9):
        """Remove highly correlated features"""
        numeric_features = df.select_dtypes(include=[np.number]).columns.tolist()

        # Remove date-related columns from correlation analysis
        numeric_features = [col for col in numeric_features if 'date' not in col.lower()]

        corr_matrix = df[numeric_features].corr().abs()
        upper_tri = corr_matrix.where(
            np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
        )

        high_corr_features = [column for column in upper_tri.columns 
                            if any(upper_tri[column] > threshold)]

        return df.drop(columns=high_corr_features, errors='ignore'), high_corr_features

    def prepare_features(self, df):
        """Prepare final feature set for training or prediction"""

        # Encode cryptocurrency names
        if 'crypto_encoded' not in df.columns:
            df['crypto_encoded'] = self.label_encoder.fit_transform(df['crypto_name'])

        # --- Ensure volatility features exist ---
        if 'volatility_14d' not in df.columns:
            df['volatility_14d'] = (
                df.groupby('crypto_name')['return_pct']
                .rolling(window=14).std()
                .reset_index(0, drop=True)
            )

        if 'gk_volatility' not in df.columns:
            df['gk_volatility'] = np.sqrt(
                0.5 * (np.log(df['high'] / df['low']))**2 -
                (2*np.log(2)-1) * (np.log(df['close'] / df['open']))**2
            )

        if 'target_volatility' not in df.columns:
            df['target_volatility'] = df['volatility_14d']

        # --- Define preferred features ---
        self.feature_columns = [
            'open', 'volume', 'marketCap', 'price_change', 'return_pct',
            'volume_change', 'marketCap_change', 'day_of_week', 'quarter',
            'volatility_14d', 'gk_volatility', 'crypto_encoded'
        ]

        # Keep only features that actually exist after correlation removal
        valid_features = [f for f in self.feature_columns if f in df.columns]

        # Remove rows where the target or critical features are NaN
        df_clean = df.dropna(subset=['target_volatility', 'volatility_14d', 'gk_volatility'])

        # Prepare X and y
        X = df_clean[valid_features]
        y = df_clean['target_volatility']

        return X, y



    def train_models(self, X, y):
        """Train multiple models and compare performance"""
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, shuffle=True
        )

        # Handle missing values
        X_train_imputed = self.imputer.fit_transform(X_train)
        X_test_imputed = self.imputer.transform(X_test)

        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train_imputed)
        X_test_scaled = self.scaler.transform(X_test_imputed)

        # Define models
        models = {
            'Linear Regression': LinearRegression(),
            'Ridge Regression': Ridge(alpha=1.0),
            'Lasso Regression': Lasso(alpha=0.01),
            'Elastic Net': ElasticNet(alpha=0.01, l1_ratio=0.5),
            'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
            'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
            'XGBoost': XGBRegressor(n_estimators=100, random_state=42),
            'SVR': SVR(kernel='rbf', C=1.0, gamma='scale')
        }

        # Train and evaluate models
        for name, model in models.items():
            print(f"Training {name}...")

            # Use scaled data for linear models, original imputed data for tree-based models
            if name in ['Linear Regression', 'Ridge Regression', 'Lasso Regression', 'Elastic Net', 'SVR']:
                model.fit(X_train_scaled, y_train)
                y_pred = model.predict(X_test_scaled)
            else:
                model.fit(X_train_imputed, y_train)
                y_pred = model.predict(X_test_imputed)

            # Calculate metrics
            r2 = r2_score(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            mae = mean_absolute_error(y_test, y_pred)

            self.results[name] = {
                'Test R²': r2,
                'Test RMSE': rmse,
                'Test MAE': mae,
                'model': model
            }

        # Select best model (highest R²)
        best_model_name = max(self.results.keys(), key=lambda k: self.results[k]['Test R²'])
        self.model = self.results[best_model_name]['model']

        print(f"\nBest Model: {best_model_name}")
        print(f"Test R²: {self.results[best_model_name]['Test R²']:.4f}")
        print(f"Test RMSE: {self.results[best_model_name]['Test RMSE']:.4f}")
        print(f"Test MAE: {self.results[best_model_name]['Test MAE']:.4f}")

        return best_model_name

    def predict(self, new_data):
        """Make predictions on new data"""
        if self.model is None:
            raise ValueError("Model not trained yet!")

        # Ensure new_data has the same features
        new_data_features = new_data[self.feature_columns]

        # Handle missing values and scale if needed
        new_data_imputed = self.imputer.transform(new_data_features)

        # Make predictions
        predictions = self.model.predict(new_data_imputed)

        return predictions

    def save_model(self, filepath='crypto_volatility_model.pkl'):
        """Save the trained model and preprocessors"""
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'imputer': self.imputer,
            'label_encoder': self.label_encoder,
            'feature_columns': self.feature_columns,
            'results': self.results
        }
        joblib.dump(model_data, filepath)
        print(f"Model saved to {filepath}")

    @classmethod
    def load_model(cls, filepath='crypto_volatility_model.pkl'):
        """Load a trained model"""
        predictor = cls()
        model_data = joblib.load(filepath)

        predictor.model = model_data['model']
        predictor.scaler = model_data['scaler']
        predictor.imputer = model_data['imputer']
        predictor.label_encoder = model_data['label_encoder']
        predictor.feature_columns = model_data['feature_columns']
        predictor.results = model_data['results']

        return predictor

def main():
    """Main training pipeline"""
    # Initialize predictor
    predictor = CryptocurrencyVolatilityPredictor()

    # Load data (replace with your dataset path)
    print("Loading data...")
    df = predictor.load_data(r"C:\Users\happy\OneDrive\ドキュメント\ml project\crypto_volatility\Crypto_volatility_prediction\data\raw\dataset.csv")

  # Update this path

    # Engineer features
    print("Engineering features...")
    df_engineered = predictor.engineer_features(df)

    # Handle outliers
    outlier_features = [
        'open', 'volume', 'marketCap', 'price_change', 'range', 
        'return_pct', 'volume_change', 'marketCap_change'
    ]
    df_clean = predictor.handle_outliers(df_engineered, outlier_features)

    # Remove highly correlated features
    df_final, dropped_features = predictor.remove_highly_correlated_features(df_clean)
    print(f"Dropped highly correlated features: {dropped_features}")

    # Prepare features
    print("Preparing features...")
    
    print("Columns available after correlation removal:", df_final.columns.tolist())
    X, y = predictor.prepare_features(df_final)
    

    
    X, y = predictor.prepare_features(df_final)


    # Train models
    print("Training models...")
    best_model = predictor.train_models(X, y)

    # Save model
    predictor.save_model()
    print("Training complete!")

if __name__ == "__main__":
    main()
